import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import timm

from torchinfo import summary
from torchview import draw_graph

import hydra
from hydra.utils import instantiate
import os
import sys

from torchmetrics.classification import F1Score

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class EfficientNetB5CutMixClassifier(pl.LightningModule):
    def __init__(self, num_classes: int = 17, lr: float = 1e-3, weight_decay: float = 1e-2, name: str = "efficientnetb5"):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.name = name
        self.embedding_dim = 128
    

        #모델 초기화 및 헤드 변경
        self.model = timm.create_model("efficientnet_b5", pretrained=True)

        # 모델 freeze
        for param in self.model.parameters():
            param.requires_grad = False

        # 마지막 레이어 헤드 초기화
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.num_classes)
        # for param in self.model.classifier.parameters():
        #   param.requires_grad = True
        

        for name, param in self.model.named_parameters():
          if (
              'blocks.5' in name              # 첫 번째 MBConv 블록
              or 'blocks.6' in name              # 마지막 MBConv 블록
              or 'conv_head' in name          # 마지막 Conv
              or 'bn2' in name                # 마지막 BN
              or 'classifier' in name         # 최종 FC
          ):
              param.requires_grad = True

        #F1 초기화
        self.train_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.train_cutmix_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        if batch["ce"] is not None:
            x, y = batch["ce"]
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(dim=1) == y).float().mean()
            preds = logits.argmax(dim=1)
             #F1 누적
            self.train_f1.update(preds, y)

            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        elif batch["cutmix"] is not None:
            x, y1, y2, lam = batch["cutmix"]
            logits = self(x)
            loss = lam * F.cross_entropy(logits, y1) + (1 - lam) * F.cross_entropy(logits, y2)
            acc = (logits.argmax(dim=1) == y1).float().mean()
            preds = logits.argmax(dim=1)
            self.train_cutmix_f1.update(preds, y1)
            self.log("train/cutmix_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/cutmix_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
            return 0.7 *loss

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
        train_cutmix_f1 = self.train_cutmix_f1.compute()
        self.log("train/f1", train_f1, prog_bar=True)
        self.log("train/cutmix_f1", train_cutmix_f1, prog_bar=True)
        self.train_f1.reset()
        self.train_cutmix_f1.reset()

    def on_validation_epoch_end(self):
        val_f1 = self.val_f1.compute()
        self.log("val/f1", val_f1, prog_bar=True)
        self.val_f1.reset()

    def predict_step(self, batch, batch_idx):
        x, img_name = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        return {"img_name": img_name, "pred": preds, "logits": logits}

    def predict_step_analyze(self, batch):
        x, img_name, label = batch

        x = x.to(self.device)
        label = label.to(self.device)

        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {"img_name": img_name, "img": x.cpu(), "pred": preds.cpu().numpy(), "label": label.cpu().numpy()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
          "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5, verbose=True, min_lr=1e-6),
          "monitor": "val/f1",
          "frequency": 1,
          "interval": "epoch",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def print_module_tree(model, indent=0):
    for name, module in model.named_children():
        print("  " * indent + f"📦 {name} → {module.__class__.__name__}")
        print_module_tree(module, indent + 1)

@hydra.main(config_path="../../configs", config_name="config")
def main(cfg):
    model = instantiate(cfg.model).model.cuda()
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    x = torch.randn(16, 3, 512, 512).cuda()
    model.eval()
    with torch.no_grad():
        out = model(x)
        print(out.shape)
    print(model.forward_features(x))

if __name__ == "__main__":
    main()