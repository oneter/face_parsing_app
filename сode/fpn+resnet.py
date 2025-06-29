# run_fpn_resnet50.py

# ... (весь код от импортов до класса Config такой же, как в run_unet_efficientnet-b2.py) ...
import os
import time
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams.update({'font.size': 10})

DATA_DIR = Path('C:/neuro_diplom/easyportrait')
CLASSES = 9
DATASET_COLORS = [[0, 0, 0], [223, 87, 188], [160, 221, 255],
                  [130, 106, 237], [200, 121, 255], [255, 183, 255],
                  [0, 144, 193], [113, 137, 255], [230, 232, 230]]


@dataclass
class Config:
    # <<< ИЗМЕНЕНИЯ ТОЛЬКО ЗДЕСЬ >>>
    MODEL_ARCH: str = 'FPN'
    ENCODER_NAME: str = 'resnet50'
    # <<< КОНЕЦ ИЗМЕНЕНИЙ >>>
    
    RUN_TRAINING: bool = True
    RESUME_FROM_CHECKPOINT: Optional[str] = None
    
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 8
    LEARNING_RATE: float = 1e-4
    MAX_EPOCHS: int = 25


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

class EasyPortraitInferDataset(Dataset):
    def __init__(self, images_dir: str, annotations_dir: str):
        self.images = sorted(glob(os.path.join(images_dir, '*')))
        self.masks = sorted(glob(os.path.join(annotations_dir, '*')))
        self.resize_transform = albu.Compose([albu.PadIfNeeded(512, 512), albu.Resize(512, 512)])
        self.normalize_transform = albu.Normalize()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        original_image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        ground_truth_mask = cv2.imread(self.masks[idx], 0)
        resized = self.resize_transform(image=original_image, mask=ground_truth_mask)
        resized_image, resized_mask = resized['image'], resized['mask']
        normalized = self.normalize_transform(image=resized_image)
        image_tensor = torch.from_numpy(normalized['image'].transpose(2, 0, 1))
        return resized_image, image_tensor, resized_mask


def get_post_augmentation():
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
        get_post_augmentation()
    ])

def get_val_test_augmentation():
    return albu.Compose([
        albu.PadIfNeeded(512, 512),
        albu.Resize(512, 512),
        get_post_augmentation()
    ])


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer_class, optimizer_params):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        image, mask = batch
        out = self(image)
        loss = self.loss_fn(out, mask.long())
        preds = torch.argmax(out, 1).unsqueeze(1)
        tp, fp, fn, tn = smp.metrics.get_stats(preds, mask.long().unsqueeze(1), mode='multiclass', num_classes=CLASSES)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        self.log(f"{stage}_loss", loss, on_step=(stage == 'train'), on_epoch=True, prog_bar=(stage == 'valid'))
        self.log(f"{stage}_iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), **self.optimizer_params)


if __name__ == "__main__":
    config = Config()
    pl.seed_everything(42, workers=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Требуется CUDA-совместимый GPU.")
    
    MODEL_ID = f"{config.MODEL_ARCH}_{config.ENCODER_NAME}"
    print(f"--- Запуск эксперимента для модели: {MODEL_ID} ---")
    
    train_dataset = EasyPortraitDataset(DATA_DIR / 'images/train', DATA_DIR / 'annotations/train', get_training_augmentation())
    val_dataset = EasyPortraitDataset(DATA_DIR / 'images/val', DATA_DIR / 'annotations/val', get_val_test_augmentation())
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, persistent_workers=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, persistent_workers=True, pin_memory=True)
    
    model = smp.create_model(config.MODEL_ARCH, encoder_name=config.ENCODER_NAME, encoder_weights="imagenet", in_channels=3, classes=CLASSES)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer_class = torch.optim.Adam
    optimizer_params = {'lr': config.LEARNING_RATE}
    lightning_model = SegmentationModel(model, loss_fn, optimizer_class, optimizer_params)
    
    best_model_path = None
    if config.RUN_TRAINING:
        print("\n--- Фаза обучения ---")
        checkpoint_dir = Path(f'./checkpoints/{MODEL_ID}')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename="best_model-{epoch:02d}-{valid_iou:.4f}", verbose=True, monitor='valid_iou', mode='max', save_top_k=1)
        tensorboard_logger = TensorBoardLogger("logs/tb_logs", name=MODEL_ID)
        csv_logger = CSVLogger("logs/csv_logs", name=MODEL_ID)
        trainer = pl.Trainer(logger=[tensorboard_logger, csv_logger], callbacks=[checkpoint_callback], precision='16-mixed', accelerator='gpu', devices=1, max_epochs=config.MAX_EPOCHS)
        trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=config.RESUME_FROM_CHECKPOINT)
        best_model_path = checkpoint_callback.best_model_path
        print(f"Обучение завершено. Лучшая модель: {best_model_path}")
    else:
        print("\n--- Обучение пропущено ---")
        if not config.RESUME_FROM_CHECKPOINT or not os.path.exists(config.RESUME_FROM_CHECKPOINT):
            raise FileNotFoundError("Не указан валидный RESUME_FROM_CHECKPOINT для теста.")
        best_model_path = config.RESUME_FROM_CHECKPOINT
        
    print(f"\n--- Фаза тестирования ---")
    print(f"Загрузка модели из: {best_model_path}")
    inference_pipeline = SegmentationModel.load_from_checkpoint(best_model_path, model=model, loss_fn=loss_fn, optimizer_class=optimizer_class, optimizer_params=optimizer_params)
    inference_model = inference_pipeline.model
    device = torch.device("cuda")
    inference_model.to(device)
    inference_model.eval()

    test_dataset = EasyPortraitInferDataset(DATA_DIR / 'images/test', DATA_DIR / 'annotations/test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)
    
    print("Прогрев модели...")
    with torch.no_grad():
        for i, (_, image_tensor, _) in enumerate(tqdm(test_dataloader, desc="Warm-up", leave=False, total=min(len(test_dataloader), 10))):
            if i >= 10: break
            _ = inference_model(image_tensor.to(device))
            
    outputs, total_loss, total_time_ms, num_images = [], 0.0, 0.0, 0
    print("Запуск тестирования с замером времени...")
    with torch.no_grad():
        for resized_img, image_tensor, gt_mask in tqdm(test_dataloader, desc="Тестирование"):
            image_tensor, gt_mask = image_tensor.to(device), gt_mask.to(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = inference_model(image_tensor)
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms += start_event.elapsed_time(end_event)
            num_images += image_tensor.shape[0]
            total_loss += loss_fn(output, gt_mask.long()).item()
            preds = torch.argmax(output, 1).unsqueeze(1)
            tp, fp, fn, tn = smp.metrics.get_stats(preds.cpu(), gt_mask.long().unsqueeze(1).cpu(), mode='multiclass', num_classes=CLASSES)
            outputs.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
            
    avg_time_ms = total_time_ms / num_images if num_images > 0 else 0
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
    tp = torch.cat([x["tp"] for x in outputs])
    fp = torch.cat([x["fp"] for x in outputs])
    fn = torch.cat([x["fn"] for x in outputs])
    tn = torch.cat([x["tn"] for x in outputs])
    
    print("\n" + "="*50)
    print("--- РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ---")
    print(f'Модель: {MODEL_ID}')
    print(f'Test Loss: {total_loss / len(test_dataloader):.4f}')
    print(f'Test IoU (micro-imagewise): {smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item():.4f}')
    print("\n--- ПРОИЗВОДИТЕЛЬНОСТЬ ---")
    print(f'Среднее время на изображение: {avg_time_ms:.2f} мс | FPS: {fps:.2f}')
    print("="*50 + "\n")
    
    per_class_metrics = np.round(torch.stack([smp.metrics.recall(tp, fp, fn, tn, reduction=None), smp.metrics.specificity(tp, fp, fn, tn, reduction=None), smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)]).numpy(), 4)
    class_names = ['Background', 'Person', 'Face skin', 'Left eyebrow', 'Right eyebrow', 'Left eye', 'Right eye', 'Lips', 'Teeth']
    metrics_data = {'Метрика': ['Recall (TPR)', 'Specificity', 'IoU']}
    for i, name in enumerate(class_names):
        metrics_data[name] = per_class_metrics[:, i]
        
    result_table_path = f'results_table_{MODEL_ID}.txt'
    with open(result_table_path, 'w', encoding='utf-8') as f:
        f.write(f"Результаты тестирования модели: {MODEL_ID}\n")
        f.write(f"Чекпоинт: {best_model_path}\n\n")
        f.write(tabulate(metrics_data, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))
    print(f"Таблица с метриками сохранена в: {result_table_path}")
    
    seed(80)
    samples_idx = sample(range(len(test_dataset)), min(len(test_dataset), 3))
    pallet = [v for c in DATASET_COLORS for v in c]
    fig, axes = plt.subplots(len(samples_idx), 3, figsize=(15, 5 * len(samples_idx)), tight_layout=True)
    if len(samples_idx) == 1: axes = np.array([axes])
    fig.suptitle(f'Примеры предсказаний модели {MODEL_ID}', fontsize=20, y=1.02)
    axes[0, 0].set_title('Исходное изображение', fontsize=14)
    axes[0, 1].set_title('Оригинальная маска', fontsize=14)
    axes[0, 2].set_title('Предсказание модели', fontsize=14)
    
    for i, idx in enumerate(samples_idx):
        image, image_tensor, gt_mask = test_dataset[idx]
        with torch.no_grad():
            prediction = inference_model(image_tensor.unsqueeze(0).to(device))
            predicted_mask_np = torch.argmax(prediction, 1).squeeze().cpu().numpy().astype(np.uint8)
        gt_mask_pil = Image.fromarray(gt_mask.astype(np.uint8)).convert('P')
        predicted_mask_pil = Image.fromarray(predicted_mask_np).convert('P')
        gt_mask_pil.putpalette(pallet)
        predicted_mask_pil.putpalette(pallet)
        for ax in axes[i, :]: ax.axis('off')
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(gt_mask_pil)
        axes[i, 2].imshow(predicted_mask_pil)
    plt.show()
