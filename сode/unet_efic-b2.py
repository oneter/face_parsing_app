import os
import time  # Импортируем для замера времени на CPU
from collections import OrderedDict
from glob import glob
from random import sample, seed
from typing import Optional

import albumentations as albu
import albumentations.pytorch as albu_pytorch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- НАСТРОЙКИ ГРАФИКОВ И КОНСТАНТЫ ---
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams.update({'font.size': 10})

DATA_DIR = 'C:/neuro_diplom/easyportrait' # ВНИМАНИЕ: Замените на свой путь

CLASSES = 9
DATASET_COLORS = [[0, 0, 0], [223, 87, 188], [160, 221, 255],
                  [130, 106, 237], [200, 121, 255], [255, 183, 255],
                  [0, 144, 193], [113, 137, 255], [230, 232, 230]]

IMGS_TRAIN_DIR = os.path.join(DATA_DIR, 'images/train')
ANNOTATIONS_TRAIN_DIR = os.path.join(DATA_DIR, 'annotations/train')
IMGS_VAL_DIR = os.path.join(DATA_DIR, 'images/val')
ANNOTATIONS_VAL_DIR = os.path.join(DATA_DIR, 'annotations/val')
IMGS_TEST_DIR = os.path.join(DATA_DIR, 'images/test')
ANNOTATIONS_TEST_DIR = os.path.join(DATA_DIR, 'annotations/test')

# --- КЛАССЫ DATASET ---
class EasyPortraitDataset(Dataset):
    def __init__(self, images_dir: str, annotations_dir: str, transform: Optional[albu.Compose] = None):
        self.images = sorted(glob(os.path.join(images_dir, '*')))
        self.masks = sorted(glob(os.path.join(annotations_dir, '*')))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], 0)

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

# Этот класс можно было бы объединить с EasyPortraitDataset, передавая ему нужный transform.
# Оставлен для сохранения исходной структуры.
class EasyPortraitInferDataset(Dataset):
    def __init__(self, images_dir: str, annotations_dir: str):
        self.images = sorted(glob(os.path.join(images_dir, '*')))
        self.masks = sorted(glob(os.path.join(annotations_dir, '*')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], 0)

        resized = albu.Compose([albu.PadIfNeeded(512, 512), albu.Resize(512, 512)])(image=image, mask=mask)
        # Для инференса мы передаем 3 элемента:
        # 1. Оригинальное изображение для визуализации
        # 2. Нормализованное изображение для подачи в модель
        # 3. Маску для расчета метрик
        normalized = albu.Normalize()(image=resized['image'])
        return resized['image'], normalized['image'].transpose(2, 0, 1), resized['mask']


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И АУГМЕНТАЦИИ ---
def visualize_seg_mask(data_sample: list, main_title: Optional[str] = None):
    num_samples = len(data_sample)
    fig, axes_list = plt.subplots(nrows=num_samples, ncols=3, figsize=(10, 5))
    plt.subplots_adjust(hspace=0, wspace=0)

    for idx in range(num_samples):
        image, mask = data_sample[idx]
        color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(DATASET_COLORS):
            color_seg[mask == label, :] = color
        masked_image = (np.array(image) * 0.5 + color_seg * 0.5).astype(np.uint8)

        for j, img in enumerate([image, mask, masked_image]):
            axes_list[idx][j].imshow(img if j != 1 else img, cmap=None if j != 1 else "gray")
            axes_list[idx][j].set_axis_off()

    if main_title:
        plt.suptitle(main_title, x=0.05, y=1.0, horizontalalignment='left',
                     fontweight='semibold', fontsize='large')
    plt.show()

def post_augmentation():
    return albu.Compose([albu.Normalize(), albu_pytorch.ToTensorV2()])

def get_training_augmentation():
    return albu.Compose([
        albu.ShiftScaleRotate(rotate_limit=25, border_mode=0, p=0.5),
        albu.PadIfNeeded(512, 512, always_apply=True, border_mode=0),
        albu.Resize(512, 512),
        albu.GaussNoise(p=0.3),
        albu.OneOf([albu.CLAHE(p=0.5), albu.RandomBrightnessContrast(p=0.6), albu.RandomGamma(p=0.4)], p=0.8),
        albu.OneOf([albu.Sharpen(p=0.5), albu.Blur(blur_limit=3, p=0.4), albu.MotionBlur(blur_limit=3, p=0.5)], p=0.7),
        albu.HueSaturationValue(p=0.15),
        post_augmentation()
    ])

def get_val_test_augmentation():
    return albu.Compose([
        albu.PadIfNeeded(512, 512),
        albu.Resize(512, 512),
        post_augmentation()
    ])


# --- МОДУЛЬ PYTORCH LIGHTNING ---
class EasyPortraitPipelines(pl.LightningModule):
    def __init__(self, model, optimizer, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_hyperparameters(ignore=['model', 'optimizer', 'criterion'])

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        image, mask = batch
        out = self(image)
        loss = self.criterion(out, mask.long())
        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(out, 1).unsqueeze(1),
                                               mask.long().unsqueeze(1),
                                               mode='multiclass',
                                               num_classes=CLASSES)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        self.log(f"{stage}_iou", iou, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=True)
        return {"loss": loss, "iou": iou}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        return self.optimizer


# --- ОСНОВНОЙ СКРИПТ ---
if __name__ == "__main__":
    # Установка seed для воспроизводимости
    pl.seed_everything(42, workers=True)

    # --- 1. ПОДГОТОВКА ДАННЫХ И МОДЕЛИ ДЛЯ ОБУЧЕНИЯ ---
    train_dataset = EasyPortraitDataset(IMGS_TRAIN_DIR, ANNOTATIONS_TRAIN_DIR, get_training_augmentation())
    val_dataset = EasyPortraitDataset(IMGS_VAL_DIR, ANNOTATIONS_VAL_DIR, get_val_test_augmentation())

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, persistent_workers=True)

    DECODER_NAME = 'UNet'
    ENCODER_NAME = 'efficientnet-b2'

    model = smp.create_model(DECODER_NAME, encoder_name=ENCODER_NAME,
                             encoder_weights="imagenet", in_channels=3, classes=CLASSES)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'./checkpoints_{DECODER_NAME}', filename=DECODER_NAME,
        verbose=True, monitor='val_loss', mode='min', save_top_k=1,
        every_n_epochs=1, save_last=False
    )

    lightning_model = EasyPortraitPipelines(model, optimizer, criterion)

    # --- 2. ОБУЧЕНИЕ МОДЕЛИ ---
    # Закомментируйте, если хотите пропустить обучение и сразу перейти к инференсу
    # trainer = pl.Trainer(
    #     callbacks=[checkpoint_callback], precision=16, accelerator='gpu', devices=1,
    #     max_epochs=21,
    #     # resume_from_checkpoint='./checkpoints_UNet/UNet.ckpt' # Используйте для продолжения обучения
    # )
    # trainer.fit(lightning_model, train_dataloader, val_dataloader)

    # =========================================================================
    # --- 3. ИНФЕРЕНС, ОЦЕНКА КАЧЕСТВА И ПРОИЗВОДИТЕЛЬНОСТИ ---
    # =========================================================================
    print("\n--- Starting Inference and Performance Evaluation ---")

    # Определяем устройство и перемещаем модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = EasyPortraitInferDataset(IMGS_TEST_DIR, ANNOTATIONS_TEST_DIR)
    # Используем batch_size=1 для честного замера времени на одном изображении
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    inference_model = smp.create_model(DECODER_NAME, encoder_name=ENCODER_NAME,
                                       encoder_weights="imagenet", in_channels=3, classes=CLASSES)

    # Загружаем лучший чекпоинт, сохраненный во время обучения
    best_model_path = 'UNet.ckpt'
    if not os.path.exists(best_model_path):
        print(f"Warning: Best checkpoint not found at '{best_model_path}'.")
        print("Please ensure training was run or provide a valid path.")
        exit()

    print(f"Loading model from: {best_model_path}")
    state_dict = torch.load(best_model_path)['state_dict']
    inference_model.load_state_dict(OrderedDict((k[6:], v) for k, v in state_dict.items()))
    inference_model.to(device)
    inference_model.eval()

    # "Прогрев" модели для стабилизации времени выполнения на GPU
    print("Warming up the model...")
    with torch.no_grad():
        for i, (_, norm_img, _) in enumerate(tqdm(test_dataloader, desc="Warm-up", leave=False)):
            if i >= 5: break # 5 итераций достаточно для прогрева
            _ = inference_model(norm_img.to(device))

    # Инициализация переменных для сбора метрик
    outputs = []
    test_loss = 0.0
    total_inference_time_ms = 0.0
    num_images = 0

    print("Starting timed inference run...")
    with torch.no_grad():
        for original_img, norm_img, mask in tqdm(test_dataloader, desc="Testing"):
            norm_img = norm_img.to(device)
            mask = mask.to(device)

            # Корректный замер времени инференса
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = inference_model(norm_img)
                end_event.record()
                torch.cuda.synchronize() # Ждем завершения операции
                batch_time_ms = start_event.elapsed_time(end_event)
            else: # для CPU
                start_time = time.perf_counter()
                output = inference_model(norm_img)
                end_time = time.perf_counter()
                batch_time_ms = (end_time - start_time) * 1000

            total_inference_time_ms += batch_time_ms
            num_images += norm_img.shape[0]

            # Считаем метрики качества
            tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(output, 1).unsqueeze(1),
                                                   mask.long().unsqueeze(1), mode='multiclass', num_classes=CLASSES)
            outputs.append({"tp": tp.cpu(), "fp": fp.cpu(), "fn": fn.cpu(), "tn": tn.cpu()})
            test_loss += criterion(output, mask.long()).item()

    # Расчет и вывод итоговых результатов
    avg_time_ms = total_inference_time_ms / num_images
    fps = 1000.0 / avg_time_ms

    tp = torch.cat([x["tp"] for x in outputs])
    fp = torch.cat([x["fp"] for x in outputs])
    fn = torch.cat([x["fn"] for x in outputs])
    tn = torch.cat([x["tn"] for x in outputs])

    print("\n--- Inference Results ---")
    print(f'Test Loss: {test_loss / len(test_dataloader):.4f}')
    print(f'IoU (micro-imagewise): {smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item():.4f}')
    print("\n--- Performance ---")
    print(f'Total images processed: {num_images}')
    print(f'Average inference time per image: {avg_time_ms:.2f} ms')
    print(f'Frames Per Second (FPS): {fps:.2f}')
    print("-----------------------\n")

    # --- 4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ НА СЛУЧАЙНЫХ ПРИМЕРАХ ---
    seed(80)
    samples = sample(range(len(test_dataset)), 2)
    pallet = [v for c in DATASET_COLORS for v in c]

    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 10), sharex='row', sharey='row',
                             subplot_kw={'xticks': [], 'yticks': []}, tight_layout=True)
    for ax, col in zip(axes[0], ['Input Image', 'Ground Truth Mask', 'Prediction']):
        ax.set_title(col, fontsize=20)

    for i, sample_idx in enumerate(samples):
        image, norm_img, mask = test_dataset[sample_idx]
        # Перемещаем тензор на device для предсказания
        input_tensor = torch.from_numpy(norm_img).unsqueeze(0).to(device)
        prediction_logits = inference_model(input_tensor)
        prediction = torch.argmax(prediction_logits, 1)

        mask_img = Image.fromarray(mask).convert('P')
        # Перемещаем результат на CPU для конвертации в NumPy/PIL
        predicted_mask_img = Image.fromarray(np.array(prediction.squeeze(0).cpu(), np.uint8)).convert('P')
        mask_img.putpalette(pallet)
        predicted_mask_img.putpalette(pallet)

        axes[i][0].imshow(image)
        axes[i][1].imshow(mask_img)
        axes[i][2].imshow(predicted_mask_img)
    plt.show()

    # --- 5. СОХРАНЕНИЕ ТАБЛИЦЫ С ПОДРОБНЫМИ МЕТРИКАМИ ---
    metrics = np.round(torch.stack([
        smp.metrics.recall(tp, fp, fn, tn, reduction="none"),
        smp.metrics.false_positive_rate(tp, fp, fn, tn, reduction="none"),
        smp.metrics.false_negative_rate(tp, fp, fn, tn, reduction="none"),
        smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
    ]).numpy(), 3)

    info = {'Metrics': ['Recall', 'FPR', 'FNR', 'IoU'],
            'Background': metrics[:, 0], 'Person': metrics[:, 1], 'Face skin': metrics[:, 2],
            'Left eyebrow': metrics[:, 3], 'Right eyebrow': metrics[:, 4], 'Left eye': metrics[:, 5],
            'Right eye': metrics[:, 6], 'Lips': metrics[:, 7], 'Teeth': metrics[:, 8]}

    with open('result_table.txt', 'w') as f:
        f.write(tabulate(info, headers='keys', tablefmt='fancy_grid'))
    print("Detailed metrics saved to result_table.txt")