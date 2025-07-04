import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18
import hydra
from hydra.utils import instantiate
import os
import sys

from torchmetrics.classification import F1Score

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes: int = 17, lr: float = 1e-3, name: str = "resnet"):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.name = name

        #모델 초기화 및 헤드 변경
        self.model = resnet18(pretrained=True)

        # 모델 freeze
        for param in self.model.parameters():
            param.requires_grad = False

        # 마지막 레이어 헤드 초기화
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        for param in self.model.fc.parameters():
          param.requires_grad = True
        
        # layer4도 학습 가능하게
        for param in self.model.layer4.parameters():
          param.requires_grad = True

        #F1 초기화
        self.train_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        preds = logits.argmax(dim=1)

        #F1 누적
        self.train_f1.update(preds, y)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        val_acc = (logits.argmax(dim=1) == y).float().mean()
        preds = logits.argmax(dim=1)
        
        #F1 누적
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)

@hydra.main(config_path="../../configs", config_name="config")
def main(cfg):
    model = instantiate(cfg.model)

if __name__ == "__main__":
    main()