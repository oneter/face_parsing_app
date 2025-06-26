import os
import time
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
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from tabulate import tabulate
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams.update({'font.size': 10})

DATA_DIR = 'C:/neuro_diplom/easyportrait'

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], 0)
        resized = albu.Compose([albu.PadIfNeeded(512, 512), albu.Resize(512, 512)])(image=image, mask=mask)
        normalized = albu.Normalize()(image=resized['image'])
        return resized['image'], normalized['image'].transpose(2, 0, 1), resized['mask']


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
        loss = self.criterion(out.float(), mask.long())
        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(out, 1).unsqueeze(1), mask.long().unsqueeze(1), mode='multiclass', num_classes=CLASSES)
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


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    
    SKIP_TRAINING = False 
    
    CHECKPOINT_PATH = './checkpoints/FPN_efficientnet-b0/best_model.ckpt'
    
    DECODER_NAME = 'FPN'
    ENCODER_NAME = 'efficientnet-b0'
    MODEL_NAME = f"{DECODER_NAME}_{ENCODER_NAME}"
    
    model_path_to_load = ""
    criterion = torch.nn.CrossEntropyLoss()

    if not SKIP_TRAINING:
        print(f"--- Preparing model for training: {MODEL_NAME} ---")
        train_dataset = EasyPortraitDataset(IMGS_TRAIN_DIR, ANNOTATIONS_TRAIN_DIR, get_training_augmentation())
        val_dataset = EasyPortraitDataset(IMGS_VAL_DIR, ANNOTATIONS_VAL_DIR, get_val_test_augmentation())
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)

        model = smp.create_model(DECODER_NAME, encoder_name=ENCODER_NAME, encoder_weights="imagenet", in_channels=3, classes=CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f'./checkpoints/{MODEL_NAME}', filename='best_model',
            verbose=True, monitor='val_loss', mode='min', save_top_k=1
        )
        
        log_dir = os.path.join("lightning_logs", MODEL_NAME)
        version_num = 0
        path_to_resume = CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None
        
        if os.path.exists(log_dir):
            existing_versions = [d for d in os.listdir(log_dir) if d.startswith("version_")]
            if existing_versions:
                max_version = max([int(v.split('_')[1]) for v in existing_versions])
                if path_to_resume:
                    version_num = max_version
                else:
                    version_num = max_version + 1
        
        print(f"Logging to version: {version_num}")
        tensorboard_logger = TensorBoardLogger("lightning_logs", name=MODEL_NAME, version=version_num)
        csv_logger = CSVLogger("csv_logs", name=MODEL_NAME, version=version_num)
        
        lightning_model = EasyPortraitPipelines(model, optimizer, criterion)

        print("\n--- Training Step ---")
        trainer = pl.Trainer(
            logger=[tensorboard_logger, csv_logger], 
            callbacks=[checkpoint_callback],
            precision="16-mixed", 
            accelerator='gpu', 
            devices=1, 
            max_epochs=35,
        )
        
        if path_to_resume:
            print(f"Resuming training from: {path_to_resume}")
        else:
            print("Starting training from scratch.")
            
        trainer.fit(
            model=lightning_model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader, 
            ckpt_path=path_to_resume
        )
        
        model_path_to_load = checkpoint_callback.best_model_path
        print("Training finished. Using best model for inference.")
    else:
        model_path_to_load = CHECKPOINT_PATH
        print("Skipping training. Loading model directly for inference.")


    print("\n--- Inference and Performance Evaluation Step ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataset = EasyPortraitInferDataset(IMGS_TEST_DIR, ANNOTATIONS_TEST_DIR)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    inference_model = smp.create_model(DECODER_NAME, encoder_name=ENCODER_NAME, encoder_weights=None, in_channels=3, classes=CLASSES)

    if not os.path.exists(model_path_to_load):
        print(f"FATAL: Checkpoint not found at '{model_path_to_load}'.")
        print("Please provide a valid path in CHECKPOINT_PATH or run the training first.")
        exit()

    print(f"Loading model from: {model_path_to_load}")
    state_dict = torch.load(model_path_to_load)['state_dict']
    inference_model.load_state_dict(OrderedDict((k[6:], v) for k, v in state_dict.items()))
    inference_model.to(device)
    inference_model.eval()

    with torch.no_grad():
        print("Warming up the model...")
        warmup_iters = min(10, len(test_dataloader))
        for i, (_, norm_img, _) in enumerate(tqdm(test_dataloader, desc="Warm-up", leave=False, total=warmup_iters)):
            if i >= warmup_iters: break
            with autocast(device_type=device.type, dtype=torch.float16):
                 _ = inference_model(norm_img.to(device))

        outputs, test_loss, total_inference_time_ms, num_images = [], 0.0, 0.0, 0
        print("Starting timed inference run...")
        for _, norm_img, mask in tqdm(test_dataloader, desc="Testing"):
            norm_img, mask = norm_img.to(device), mask.to(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with autocast(device_type=device.type, dtype=torch.float16):
                output = inference_model(norm_img)
            end_event.record()
            torch.cuda.synchronize()
            batch_time_ms = start_event.elapsed_time(end_event)

            total_inference_time_ms += batch_time_ms
            num_images += norm_img.shape[0]
            
            test_loss += criterion(output.float(), mask.long()).item()
            tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(output.float(), 1).unsqueeze(1), mask.long().unsqueeze(1), mode='multiclass', num_classes=CLASSES)
            outputs.append({"tp": tp.cpu(), "fp": fp.cpu(), "fn": fn.cpu(), "tn": tn.cpu()})
            
    avg_time_ms = total_inference_time_ms / num_images if num_images > 0 else 0
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
    tp, fp, fn, tn = torch.cat([x["tp"] for x in outputs]), torch.cat([x["fp"] for x in outputs]), torch.cat([x["fn"] for x in outputs]), torch.cat([x["tn"] for x in outputs])

    print("\n" + "="*50)
    print("--- INFERENCE RESULTS ---")
    print(f'Model: {MODEL_NAME}')
    print(f'Test Loss: {test_loss / len(test_dataloader):.4f}')
    print(f'IoU (micro-imagewise): {smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item():.4f}')
    print("\n--- PERFORMANCE ---")
    print(f'Average inference time per image: {avg_time_ms:.2f} ms')
    print(f'Frames Per Second (FPS): {fps:.2f}')
    print("="*50 + "\n")

    print("Calculating and saving detailed metrics to result_table.txt...")
    metrics = np.round(torch.stack([
        smp.metrics.recall(tp, fp, fn, tn, reduction="none"),
        smp.metrics.false_positive_rate(tp, fp, fn, tn, reduction="none"),
        smp.metrics.false_negative_rate(tp, fp, fn, tn, reduction="none"),
        smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
    ]).numpy(), 3)

    info = {'Metrics':     ['Recall', 'FPR', 'FNR', 'IoU'],
            'Background':  metrics[:, 0].tolist(),
            'Person':      metrics[:, 1].tolist(),
            'Face skin':   metrics[:, 2].tolist(),
            'Left eyebrow':metrics[:, 3].tolist(),
            'Right eyebrow':metrics[:, 4].tolist(),
            'Left eye':    metrics[:, 5].tolist(),
            'Right eye':   metrics[:, 6].tolist(),
            'Lips':        metrics[:, 7].tolist(),
            'Teeth':       metrics[:, 8].tolist()}

    with open(f'result_table_{MODEL_NAME}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Detailed metrics for model: {MODEL_NAME}\n\n")
        f.write(tabulate(info, headers='keys', tablefmt='fancy_grid'))
    print(f"Metrics saved to result_table_{MODEL_NAME}.txt")

    print("Visualizing some predictions...")
    seed(80) 
    samples_to_show = sample(range(len(test_dataset)), min(len(test_dataset), 3))
    pallet = [v for c in DATASET_COLORS for v in c]

    fig, axes = plt.subplots(len(samples_to_show), 3, figsize=(15, 5 * len(samples_to_show)), 
                             sharex=True, sharey=True, subplot_kw={'xticks': [], 'yticks': []})
    fig.tight_layout()
    if len(samples_to_show) == 1:
        axes = np.array([axes])

    for ax, col in zip(axes[0], ['Input Image', 'Ground Truth Mask', 'Prediction']):
        ax.set_title(col, fontsize=16)

    for i, sample_idx in enumerate(samples_to_show):
        image, norm_img, mask = test_dataset[sample_idx]
        input_tensor = torch.from_numpy(norm_img).unsqueeze(0).to(device)
        with torch.no_grad(), autocast(device_type=device.type, dtype=torch.float16):
            prediction_logits = inference_model(input_tensor)
        prediction = torch.argmax(prediction_logits, 1)

        mask_img = Image.fromarray(mask).convert('P')
        predicted_mask_img = Image.fromarray(np.array(prediction.squeeze(0).cpu(), np.uint8)).convert('P')
        mask_img.putpalette(pallet)
        predicted_mask_img.putpalette(pallet)

        axes[i][0].imshow(image)
        axes[i][1].imshow(mask_img)
        axes[i][2].imshow(predicted_mask_img)
    plt.show()