import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import AutoModel, AutoImageProcessor
from torchmetrics.classification import F1Score

class DINOv2Classifier(pl.LightningModule):
    def __init__(self, num_classes: int = 17, lr: float = 1e-4, weight_decay: float = 1e-2, model_name: str = "facebook/dinov2-base"):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_name = model_name

        # DINOv2 모델 로드
        self.feature_extractor = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Feature dim은 dinov2-base 기준 768 (dinov2-large는 1024)
        feature_dim = self.feature_extractor.config.hidden_size

        for name, param in self.feature_extractor.named_parameters():
            if any(f"layer.{i}" in name for i in [10, 11]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # Metric
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        features = self.feature_extractor(pixel_values=x).last_hidden_state[:, 0, :]  # CLS 토큰
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        preds = logits.argmax(dim=1)
        self.train_f1.update(preds, y)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        preds = logits.argmax(dim=1)
        self.val_f1.update(preds, y)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/f1", self.train_f1.compute(), prog_bar=True)
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5, min_lr=1e-6),
            "monitor": "val/f1",
            "interval": "epoch",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
