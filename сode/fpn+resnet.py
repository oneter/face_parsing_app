import os
from collections import OrderedDict
from glob import glob
from random import seed, sample
from typing import Optional

import albumentations as albu
import albumentations.pytorch as albu_pytorch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- НАСТРОЙКИ ГРАФИКОВ ---
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams.update({'font.size': 10})

# --- КОНСТАНТЫ ---
DATA_DIR = 'C:/neuro_diplom/easyportrait'
CLASSES = 9
DATASET_COLORS = [[0, 0, 0], [223, 87, 188], [160, 221, 255],
                  [130, 106, 237], [200, 121, 255], [255, 183, 255],
                  [0, 144, 193], [113, 137, 255], [230, 232, 230]]

# --- ПУТИ К ДАННЫМ ---
IMGS_TRAIN_DIR = os.path.join(DATA_DIR, 'images/train')
ANNOTATIONS_TRAIN_DIR = os.path.join(DATA_DIR, 'annotations/train')
IMGS_VAL_DIR = os.path.join(DATA_DIR, 'images/val')
ANNOTATIONS_VAL_DIR = os.path.join(DATA_DIR, 'annotations/val')
IMGS_TEST_DIR = os.path.join(DATA_DIR, 'images/test')
ANNOTATIONS_TEST_DIR = os.path.join(DATA_DIR, 'annotations/test')

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

def post_augmentation():
    return albu.Compose([albu.Normalize(), albu_pytorch.ToTensorV2()])

def get_training_augmentation():
    return albu.Compose([
        albu.ShiftScaleRotate(rotate_limit=25, border_mode=0, p=0.5),
        albu.PadIfNeeded(512, 512, always_apply=True, border_mode=0),
        albu.Resize(512, 512),
        albu.GaussNoise(p=0.3),
        albu.OneOf([
            albu.CLAHE(p=0.5),
            albu.RandomBrightnessContrast(p=0.6),
            albu.RandomGamma(p=0.4),
        ], p=0.8),
        albu.OneOf([
            albu.Sharpen(p=0.5),
            albu.Blur(blur_limit=3, p=0.4),
            albu.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.7),
        albu.HueSaturationValue(p=0.15),
        post_augmentation()
    ])

def get_val_test_augmentation():
    return albu.Compose([
        albu.PadIfNeeded(512, 512),
        albu.Resize(512, 512),
        post_augmentation()
    ])

class EasyPortraitPipelines(pl.LightningModule):
    def __init__(self, model, optimizer_class, optimizer_params, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.save_hyperparameters(ignore=['model', 'criterion'])

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        image, mask = batch
        out = self(image)
        loss = self.criterion(out, mask.long())
        
        preds = torch.argmax(out, 1).unsqueeze(1)
        
        tp, fp, fn, tn = smp.metrics.get_stats(preds,
                                               mask.long().unsqueeze(1),
                                               mode='multiclass',
                                               num_classes=CLASSES)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        
        self.log(f"{stage}_loss", loss, on_step=(stage == 'train'), on_epoch=True, prog_bar=(stage == 'valid'))
        self.log(f"{stage}_iou", iou, on_step=(stage == 'train'), on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), **self.optimizer_params)

class EasyPortraitInferDataset(Dataset):
    def __init__(self, images_dir: str, annotations_dir: str):
        self.images = sorted(glob(os.path.join(images_dir, '*')))
        self.masks = sorted(glob(os.path.join(annotations_dir, '*')))
        
        self.resize_transform = albu.Compose([albu.PadIfNeeded(512, 512), albu.Resize(512, 512)])
        self.normalize_transform = albu.Normalize()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask_gt = cv2.imread(self.masks[idx], 0)

        resized = self.resize_transform(image=image, mask=mask_gt)
        resized_image, resized_mask = resized['image'], resized['mask']
        
        normalized = self.normalize_transform(image=resized_image)
        normalized_image_tensor = torch.from_numpy(normalized['image'].transpose(2, 0, 1))

        return resized_image, normalized_image_tensor, resized_mask

def plot_and_save_metrics(log_dir: str, file_name: str):
    metrics_path = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        print(f"Файл с метриками не найден: {metrics_path}")
        return

    metrics_df = pd.read_csv(metrics_path)
    agg_metrics = metrics_df.groupby('epoch').last().reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Прогресс обучения модели', fontsize=16)

    ax1.plot(agg_metrics['epoch'], agg_metrics.get('train_loss'), label='Train Loss', marker='.')
    ax1.plot(agg_metrics['epoch'], agg_metrics.get('valid_loss'), label='Validation Loss', marker='o')
    ax1.set_ylabel('Loss')
    ax1.set_title('Динамика Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(agg_metrics['epoch'], agg_metrics.get('train_iou'), label='Train IoU', marker='.')
    ax2.plot(agg_metrics['epoch'], agg_metrics.get('valid_iou'), label='Validation IoU', marker='o')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('IoU (Intersection over Union)')
    ax2.set_title('Динамика IoU')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(file_name)
    print(f"Графики обучения сохранены в файл: {file_name}")
    plt.show()

# --- ОСНОВНОЙ БЛОК ИСПОЛНЕНИЯ ---
# Эта конструкция ОБЯЗАТЕЛЬНА для использования multiprocessing (num_workers > 0) в Windows.
# Она предотвращает рекурсивный запуск кода в дочерних процессах.
if __name__ == "__main__":
    
    # --- 1. ПРОВЕРКА НАЛИЧИЯ GPU ---
    # Принудительно завершаем работу, если нет доступного GPU
    if not torch.cuda.is_available():
        raise RuntimeError("Требуется наличие CUDA-совместимого GPU для выполнения этого скрипта.")
    
    # --- 2. НАСТРОЙКИ МОДЕЛИ И ОБУЧЕНИЯ ---
    DECODER_NAME = 'FPN'
    ENCODER_NAME = 'resnet50'
    MODEL_ID = f"{DECODER_NAME}_{ENCODER_NAME}"
    BATCH_SIZE = 8
    # Установите количество воркеров в зависимости от количества ядер вашего CPU (разумно - 4, 8)
    NUM_WORKERS = 6 

    print(f"Начало работы с моделью: {MODEL_ID}")
    print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
    print(f"Batch Size: {BATCH_SIZE}, Num Workers: {NUM_WORKERS}")

    # --- 3. ПОДГОТОВКА ДАННЫХ ---
    train_dataset = EasyPortraitDataset(IMGS_TRAIN_DIR, ANNOTATIONS_TRAIN_DIR, get_training_augmentation())
    val_dataset = EasyPortraitDataset(IMGS_VAL_DIR, ANNOTATIONS_VAL_DIR, get_val_test_augmentation())

    # Активируем num_workers и persistent_workers для ускорения загрузки данных
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=NUM_WORKERS, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=NUM_WORKERS, persistent_workers=True)

    # --- 4. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ---
    model = smp.create_model(DECODER_NAME, encoder_name=ENCODER_NAME,
                             encoder_weights="imagenet", in_channels=3, classes=CLASSES)

    optimizer_class = torch.optim.Adam
    optimizer_params = {'lr': 1e-4}
    criterion = torch.nn.CrossEntropyLoss()

    lightning_model = EasyPortraitPipelines(model, optimizer_class, optimizer_params, criterion)
    
    # --- 5. НАСТРОЙКА ОБУЧЕНИЯ ---
    checkpoint_dir = f'./checkpoints_{MODEL_ID}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=MODEL_ID + "-{epoch:02d}-{valid_loss:.2f}",
        verbose=True,
        monitor='valid_loss',
        mode='min',
        save_top_k=1,
    )

    logger = CSVLogger("logs", name=MODEL_ID)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        precision='16-mixed',
        accelerator='gpu',
        devices=1,
        max_epochs=25,
    )

    # --- 6. ОБУЧЕНИЕ ---
    trainer.fit(lightning_model, train_dataloader, val_dataloader)
    
    # --- 7. АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ ---
    plot_and_save_metrics(trainer.logger.log_dir, f'training_progress_{MODEL_ID}.png')

    # --- 8. ТЕСТИРОВАНИЕ И ОЦЕНКА (INFERENCE) ---
    print("\n--- Начало тестирования лучшей модели ---")
    
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        print("Не удалось найти лучший чекпоинт. Тестирование прервано.")
        exit()
        
    print(f"Загрузка лучшей модели из: {best_model_path}")

    inference_model = EasyPortraitPipelines.load_from_checkpoint(best_model_path).model
    
    device = torch.device("cuda") # Мы уже проверили наличие GPU
    inference_model.to(device)
    inference_model.eval()

    test_dataset = EasyPortraitInferDataset(IMGS_TEST_DIR, ANNOTATIONS_TEST_DIR)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    outputs = []
    test_loss_sum = 0.0
    with torch.no_grad():
        for _, norm_img_tensor, mask_gt in tqdm(test_dataloader, desc="Тестирование"):
            norm_img_tensor = norm_img_tensor.to(device)
            mask_gt = mask_gt.to(device)

            output = inference_model(norm_img_tensor)
            
            preds = torch.argmax(output, 1).unsqueeze(1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds.cpu(), mask_gt.long().unsqueeze(1).cpu(), 
                mode='multiclass', num_classes=CLASSES
            )
            outputs.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
            test_loss_sum += criterion(output, mask_gt.long()).item()

    tp = torch.cat([x["tp"] for x in outputs])
    fp = torch.cat([x["fp"] for x in outputs])
    fn = torch.cat([x["fn"] for x in outputs])
    tn = torch.cat([x["tn"] for x in outputs])

    print(f'\nTest Loss: {test_loss_sum / len(test_dataloader):.4f}')
    print(f'Test IoU (micro-imagewise): {smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item():.4f}')

    # ... (остальной код для визуализации и сохранения таблицы остается без изменений) ...

    seed(80)
    samples_idx = sample(range(len(test_dataset)), 2)
    pallet = [v for c in DATASET_COLORS for v in c]

    fig, axes = plt.subplots(len(samples_idx), 3, figsize=(15, 10), tight_layout=True)
    fig.suptitle(f'Примеры предсказаний модели {MODEL_ID}', fontsize=20)
    
    axes_flat = axes.flatten()
    axes_flat[0].set_title('Исходное изображение', fontsize=14)
    axes_flat[1].set_title('Оригинальная маска', fontsize=14)
    axes_flat[2].set_title('Предсказание модели', fontsize=14)

    for i, idx in enumerate(samples_idx):
        image, norm_img_tensor, mask_gt = test_dataset[idx]
        
        with torch.no_grad():
            prediction = inference_model(norm_img_tensor.unsqueeze(0).to(device))
            predicted_mask_np = torch.argmax(prediction, 1).squeeze().cpu().numpy().astype(np.uint8)

        mask_gt_pil = Image.fromarray(mask_gt.astype(np.uint8)).convert('P')
        predicted_mask_pil = Image.fromarray(predicted_mask_np).convert('P')
        
        mask_gt_pil.putpalette(pallet)
        predicted_mask_pil.putpalette(pallet)

        row_idx = i * 3
        axes_flat[row_idx].imshow(image)
        axes_flat[row_idx].axis('off')
        axes_flat[row_idx + 1].imshow(mask_gt_pil)
        axes_flat[row_idx + 1].axis('off')
        axes_flat[row_idx + 2].imshow(predicted_mask_pil)
        axes_flat[row_idx + 2].axis('off')

    plt.show()

    per_class_metrics = np.round(torch.stack([
        smp.metrics.recall(tp, fp, fn, tn, reduction=None),
        1 - smp.metrics.fpr(tp, fp, fn, tn, reduction=None),
        smp.metrics.fnr(tp, fp, fn, tn, reduction=None),
        smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
    ]).numpy(), 4)

    info = {
        'Метрика': ['Recall (TPR)', 'Specificity', 'FNR', 'IoU'],
        'Background': per_class_metrics[:, 0], 'Person': per_class_metrics[:, 1],
        'Face skin': per_class_metrics[:, 2], 'Left eyebrow': per_class_metrics[:, 3],
        'Right eyebrow': per_class_metrics[:, 4], 'Left eye': per_class_metrics[:, 5],
        'Right eye': per_class_metrics[:, 6], 'Lips': per_class_metrics[:, 7],
        'Teeth': per_class_metrics[:, 8]
    }

    result_table_path = f'result_table_{MODEL_ID}.txt'
    with open(result_table_path, 'w', encoding='utf-8') as f:
        f.write(f"Результаты тестирования модели: {MODEL_ID}\n")
        f.write(f"Лучший чекпоинт: {best_model_path}\n\n")
        f.write(tabulate(info, headers='keys', tablefmt='fancy_grid'))
    
    print(f"Таблица с результатами сохранена в файл: {result_table_path}")