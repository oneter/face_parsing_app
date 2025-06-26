import sys
import os
import argparse
import logging
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import albumentations as albu
import segmentation_models_pytorch as smp

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QComboBox, QFileDialog,
                             QTabWidget, QStatusBar, QMessageBox, QStyle)
from PyQt5.QtCore import (QThread, pyqtSignal, Qt, QTimer, QSize, QSettings,
                          QObject, QByteArray)
from PyQt5.QtGui import QImage, QPixmap, QIcon


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_log.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


MODEL_DECODER = 'FPN'
MODEL_ENCODER = 'efficientnet-b0'
CLASSES = 9
IMAGE_SIZE = 512


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

DEFAULT_CHECKPOINT = resource_path('best_model.ckpt')

PALETTE = np.array([
    [0, 0, 0], [188, 87, 223], [255, 221, 160], [237, 106, 130], [255, 121, 200],
    [255, 183, 255], [193, 144, 0], [255, 137, 113], [230, 232, 230]
], dtype=np.uint8)


def get_preprocessing_transform():
    return albu.Compose([
        albu.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        albu.Resize(IMAGE_SIZE, IMAGE_SIZE), albu.Normalize(),
    ])

def numpy_to_qimage(np_image):
    if np_image.ndim == 3 and np_image.shape[2] == 3:
        height, width, channel = np_image.shape
        bytes_per_line = 3 * width
        return QImage(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB).data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QImage()


class InferenceWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray, float)
    static_result_ready = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, model, device, preproc_transform):
        super().__init__()
        self.model = model
        self.device = device
        self.preproc_transform = preproc_transform

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray, opacity: float):
        try:
            start_time = time.perf_counter()
            original_shape = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = self.preproc_transform(image=rgb_frame)
            image_tensor = torch.from_numpy(processed['image']).permute(2, 0, 1).unsqueeze(0).to(self.device)

            prediction_logits = self.model(image_tensor)
            prediction_mask = torch.argmax(prediction_logits, 1).squeeze(0).cpu().numpy().astype(np.uint8)
            prediction_mask_resized = cv2.resize(prediction_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
            colored_mask = PALETTE[prediction_mask_resized]

            alpha = 1.0 - opacity
            blended_frame = cv2.addWeighted(frame, alpha, colored_mask, opacity, 0)
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            self.frame_ready.emit(blended_frame, inference_time_ms)
        except Exception as e:
            logging.error(f"Ошибка в потоке инференса: {e}", exc_info=True)

    @torch.no_grad()
    def process_static_image(self, image_path: str):
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                raise IOError(f"Не удалось прочитать изображение: {image_path}")

            original_shape = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = self.preproc_transform(image=rgb_frame)
            image_tensor = torch.from_numpy(processed['image']).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            prediction_logits = self.model(image_tensor)
            prediction_mask = torch.argmax(prediction_logits, 1).squeeze(0).cpu().numpy().astype(np.uint8)
            prediction_mask_resized = cv2.resize(prediction_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
            colored_mask = PALETTE[prediction_mask_resized]
            
            self.static_result_ready.emit(frame, colored_mask)
        except Exception as e:
            logging.error(f"Ошибка при обработке статического изображения: {e}", exc_info=True)


class MainWindow(QMainWindow):
    request_process_frame = pyqtSignal(np.ndarray, float)
    request_process_static = pyqtSignal(str)

    def __init__(self, model, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.setWindowTitle("Сегментация лица в реальном времени")
        self.setGeometry(100, 100, 1280, 720)
        self.settings = QSettings("MyCompany", "FaceParserApp")
        self.restoreGeometry(self.settings.value("geometry", QByteArray()))
        self.restoreState(self.settings.value("windowState", QByteArray()))
        
        self.inference_thread = QThread()
        self.worker = InferenceWorker(self.model, self.device, get_preprocessing_transform())
        self.worker.moveToThread(self.inference_thread)
        self.inference_thread.start()

        self.webcam_timer = QTimer(self)
        self.video_capture = None
        self.current_fps = 0
        self.webcam_opacity = 0.5
        
        self.is_webcam_running = False
        
        self.static_image_path = None
        self.source_static_image = None
        self.processed_static_mask = None

        self.init_ui()
        self.connect_signals()
        self.populate_camera_list()
        logging.info(f"Приложение запущено. Устройство: {self.device.type.upper()}")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tabs = QTabWidget()
        self.webcam_tab = QWidget()
        self.image_tab = QWidget()
        self.tabs.addTab(self.webcam_tab, "Режим вебкамеры")
        self.tabs.addTab(self.image_tab, "Режим изображения")
        main_layout.addWidget(self.tabs)

        webcam_layout = QHBoxLayout(self.webcam_tab)
        webcam_controls_layout = QVBoxLayout()
        webcam_controls_layout.setAlignment(Qt.AlignTop)
        self.camera_selector = QComboBox()
        self.start_button = QPushButton("Запустить")
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setEnabled(False)
        self.webcam_opacity_slider = QSlider(Qt.Horizontal)
        self.webcam_opacity_slider.setRange(0, 100)
        self.webcam_opacity_slider.setValue(50)
        webcam_controls_layout.addWidget(QLabel("Выберите камеру:"))
        webcam_controls_layout.addWidget(self.camera_selector)
        webcam_controls_layout.addWidget(self.start_button)
        webcam_controls_layout.addWidget(self.stop_button)
        webcam_controls_layout.addSpacing(20)
        webcam_controls_layout.addWidget(QLabel("Прозрачность маски:"))
        webcam_controls_layout.addWidget(self.webcam_opacity_slider)
        self.video_label = QLabel("Изображение с вебкамеры появится здесь.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #333;")
        self.video_label.setMinimumSize(640, 480)
        webcam_layout.addWidget(self.video_label, 3)
        webcam_layout.addLayout(webcam_controls_layout, 1)

        image_layout = QVBoxLayout(self.image_tab)
        image_controls_layout = QHBoxLayout()
        image_controls_layout.setAlignment(Qt.AlignLeft)
        self.load_image_button = QPushButton("Загрузить изображение")
        self.process_image_button = QPushButton("Обработать изображение")
        self.process_image_button.setEnabled(False)
        self.save_image_button = QPushButton("Сохранить результат")
        self.save_image_button.setEnabled(False)
        self.image_opacity_slider = QSlider(Qt.Horizontal)
        self.image_opacity_slider.setRange(0, 100)
        self.image_opacity_slider.setValue(50)
        self.image_opacity_slider.setFixedWidth(200)
        image_controls_layout.addWidget(self.load_image_button)
        image_controls_layout.addWidget(self.process_image_button)
        image_controls_layout.addWidget(self.save_image_button)
        image_controls_layout.addStretch()
        image_controls_layout.addWidget(QLabel("Прозрачность маски:"))
        image_controls_layout.addWidget(self.image_opacity_slider)

        image_display_layout = QHBoxLayout()
        self.source_image_label = QLabel("Загрузите изображение для начала.")
        self.source_image_label.setAlignment(Qt.AlignCenter)
        self.source_image_label.setStyleSheet("border: 1px solid #aaa;")
        self.result_image_label = QLabel("Результат будет показан здесь.")
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setStyleSheet("border: 1px solid #aaa;")
        image_display_layout.addWidget(self.source_image_label)
        image_display_layout.addWidget(self.result_image_label)
        image_layout.addLayout(image_controls_layout)
        image_layout.addLayout(image_display_layout, 1)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.fps_label = QLabel("Кадров/с: н/д")
        self.device_label = QLabel(f"Устройство: {self.device.type.upper()}")
        self.model_label = QLabel(f"Модель: {MODEL_DECODER}+{MODEL_ENCODER}")
        self.status_bar.addPermanentWidget(self.model_label)
        self.status_bar.addPermanentWidget(self.device_label)
        self.status_bar.addPermanentWidget(self.fps_label)

    def connect_signals(self):
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.static_result_ready.connect(self.update_static_result)
        self.request_process_frame.connect(self.worker.process_frame)
        self.request_process_static.connect(self.worker.process_static_image)
        self.webcam_timer.timeout.connect(self.process_next_frame)
        self.start_button.clicked.connect(self.start_webcam)
        self.stop_button.clicked.connect(self.stop_webcam)
        self.webcam_opacity_slider.valueChanged.connect(self.set_webcam_opacity)
        self.load_image_button.clicked.connect(self.load_image)
        self.process_image_button.clicked.connect(self.process_image)
        self.save_image_button.clicked.connect(self.save_image)
        self.image_opacity_slider.valueChanged.connect(self.update_static_blend)

    def set_webcam_opacity(self, value):
        self.webcam_opacity = value / 100.0
    
    def populate_camera_list(self):
        self.camera_selector.clear()
        index = 0
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened(): break
            self.camera_selector.addItem(f"Камера {index}")
            cap.release()
            index += 1
        if index == 0:
            self.camera_selector.addItem("Камеры не найдены")
            self.start_button.setEnabled(False)

    def start_webcam(self):
        camera_index = self.camera_selector.currentIndex()
        if camera_index < 0:
            QMessageBox.warning(self, "Предупреждение", "Камера не выбрана или недоступна.")
            return
        self.video_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.video_capture.isOpened():
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть Камеру {camera_index}.")
            self.video_capture = None
            return
        
        self.is_webcam_running = True
        
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #333;")
        self.webcam_timer.start(33)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_selector.setEnabled(False)
        logging.info(f"Трансляция с Камеры {camera_index} запущена")

    def stop_webcam(self):
        self.is_webcam_running = False
        
        self.webcam_timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_selector.setEnabled(True)
        self.video_label.clear()
        self.video_label.setText("Трансляция остановлена.")
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #000;")
        self.fps_label.setText("Кадров/с: н/д")
        logging.info("Трансляция остановлена.")

    def process_next_frame(self):
        if self.video_capture and self.is_webcam_running:
            ret, frame = self.video_capture.read()
            if ret:
                self.request_process_frame.emit(frame, self.webcam_opacity)

    def update_video_frame(self, frame, inference_time_ms):
        if not self.is_webcam_running:
            return
            
        q_image = numpy_to_qimage(frame)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.current_fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0
        self.fps_label.setText(f"Кадров/с: {self.current_fps:.2f}")
        
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Файлы изображений (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.static_image_path = path
            self.source_static_image = cv2.imread(path)
            if self.source_static_image is None:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение: {path}")
                return
            pixmap = QPixmap(path)
            self.source_image_label.setPixmap(pixmap.scaled(self.source_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.process_image_button.setEnabled(True)
            self.save_image_button.setEnabled(False)
            self.result_image_label.clear()
            self.result_image_label.setText("Нажмите 'Обработать изображение'")
            self.processed_static_mask = None
            logging.info(f"Изображение загружено: {path}")

    def process_image(self):
        if self.static_image_path:
            self.result_image_label.setText("Обработка...")
            self.image_opacity_slider.setValue(50)
            self.request_process_static.emit(self.static_image_path)
            
    def update_static_result(self, source_img, mask_img):
        self.source_static_image = source_img
        self.processed_static_mask = mask_img
        self.update_static_blend(self.image_opacity_slider.value())
        self.save_image_button.setEnabled(True)
        logging.info("Изображение успешно обработано.")

    def update_static_blend(self, value):
        if self.source_static_image is not None and self.processed_static_mask is not None:
            opacity = value / 100.0
            alpha = 1.0 - opacity
            blended_image = cv2.addWeighted(self.source_static_image, alpha, self.processed_static_mask, opacity, 0)
            q_image_result = numpy_to_qimage(blended_image)
            pixmap_result = QPixmap.fromImage(q_image_result)
            target_size = self.source_image_label.size()
            self.result_image_label.setPixmap(pixmap_result.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_image(self):
        if self.source_static_image is not None and self.processed_static_mask is not None:
            default_path = os.path.splitext(self.static_image_path)[0] + "_result.png"
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить результат", default_path, "Изображение PNG (*.png)")
            if path:
                try:
                    opacity = self.image_opacity_slider.value() / 100.0
                    alpha = 1.0 - opacity
                    blended_image = cv2.addWeighted(self.source_static_image, alpha, self.processed_static_mask, opacity, 0)
                    cv2.imwrite(path, blended_image)
                    mask_path = os.path.splitext(path)[0] + "_mask.png"
                    cv2.imwrite(mask_path, self.processed_static_mask)
                    QMessageBox.information(self, "Успех", f"Результат сохранен в:\n{path}\n\nМаска сохранена в:\n{mask_path}")
                    logging.info(f"Результат сохранен в {path} и {mask_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить изображение: {e}")
                    logging.error(f"Не удалось сохранить изображение: {e}")

    def closeEvent(self, event):
        logging.info("Закрытие приложения...")
        self.stop_webcam()
        self.inference_thread.quit()
        self.inference_thread.wait()
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        logging.info("Приложение закрыто.")
        event.accept()


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        logging.error(f"КРИТИЧЕСКАЯ ОШИБКА: Чекпоинт не найден по пути '{checkpoint_path}'.")
        return None
    try:
        logging.info(f"Загрузка модели из: {checkpoint_path}")
        model = smp.create_model(MODEL_DECODER, encoder_name=MODEL_ENCODER, encoder_weights=None, in_channels=3, classes=CLASSES)
        state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
        clean_state_dict = OrderedDict((k[6:], v) for k, v in state_dict.items())
        model.load_state_dict(clean_state_dict)
        model.to(device)
        model.eval()
        logging.info("Модель успешно загружена.")
        return model
    except Exception as e:
        logging.error(f"Не удалось загрузить модель: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Графическое приложение для сегментации лица в реальном времени")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Путь к файлу с чекпоинтом модели.')
    args = parser.parse_args()
    app = QApplication(sys.argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    if model is None:
        error_msg = f"Не удалось загрузить модель из '{args.checkpoint}'.\nУбедитесь, что файл существует и является корректным чекпоинтом."
        QMessageBox.critical(None, "Ошибка загрузки модели", error_msg)
        sys.exit(1)
    main_window = MainWindow(model, device)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()