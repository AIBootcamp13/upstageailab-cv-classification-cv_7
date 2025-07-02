import hydra
import os
import sys

from pytorch_lightning.callbacks import EarlyStopping

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from hydra.utils import instantiate
from src.dataset.datamodule import DocumentDataModule

@hydra.main(config_path="../../configs", config_name="config")
def train(cfg):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",   # 기준 지표 (validation loss 또는 val_acc 등)
        mode="min",           # 'min'일 경우 loss가 줄어드는 방향으로, 'max'는 acc처럼 클수록 좋은 지표에 사용
        patience=3,           # 성능이 개선되지 않는 epoch 수
        verbose=True
    )
    dm = DocumentDataModule(**cfg.data)
    model = instantiate(cfg.model)
    trainer = Trainer(**cfg.train, callbacks=[early_stop_callback])
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    train()
