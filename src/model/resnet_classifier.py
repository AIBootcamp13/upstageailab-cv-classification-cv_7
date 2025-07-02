import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18
import hydra
from hydra.utils import instantiate
import os
import sys

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes: int = 17, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss)
        self.log("train/acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        val_acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", val_loss)
        self.log("val/acc", val_acc)

    def predict_step(self, batch, batch_idx):
        x, img_name = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {"img_name": img_name, "pred": preds}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

@hydra.main(config_path="../../configs", config_name="config")
def main(cfg):
    model = instantiate(cfg.model)
    print(model)

if __name__ == "__main__":
    main()