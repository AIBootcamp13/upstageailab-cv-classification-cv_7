import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import timm

from torchmetrics.classification import F1Score

class TimmClassifier(pl.LightningModule):
    def __init__(self, name: str, num_classes: int = 17, lr: float = 1e-3, weight_decay = 1e-4, dropout = 0.3):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = name
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.model = timm.create_model(
            name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=self.dropout
        )

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.train_f1.update(preds, y)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        val_acc = (preds == y).float().mean()

        self.val_f1.update(preds, y)

        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", val_acc, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss

    def on_train_epoch_end(self):
        train_f1 = self.train_f1.compute()
        self.log("train/f1", train_f1, prog_bar=True)
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        val_f1 = self.val_f1.compute()
        self.log("val/f1", val_f1, prog_bar=True)
        self.val_f1.reset()

    def predict_step(self, batch, batch_idx):
        x, img_name = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {"img_name": img_name, "pred": preds, "logits": logits}

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
