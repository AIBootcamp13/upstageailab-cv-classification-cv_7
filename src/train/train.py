import hydra
import os
import sys
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from hydra.utils import instantiate
from src.dataset.datamodule import DocumentDataModule

@hydra.main(config_path="../../configs", config_name="config")
def train(cfg):
    wandb.login()

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",   # 기준 지표 (validation loss 또는 val_acc 등)
        mode="min",           # 'min'일 경우 loss가 줄어드는 방향으로, 'max'는 acc처럼 클수록 좋은 지표에 사용
        patience=3,           # 성능이 개선되지 않는 epoch 수
        verbose=True
    )

    dm = DocumentDataModule(**cfg.data)
    model = instantiate(cfg.model)
    trainer = Trainer(**cfg.train, callbacks=[early_stop_callback], logger=wandb_logger)
    trainer.fit(model, datamodule=dm)

    # 모델 저장
    model_path = os.path.join(ROOT_DIR, "artifacts", f"{cfg.model.name}.pt")
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact(name = f"{cfg.model.name}", type="model")
    artifact.add_file(model_path)
    wandb_logger.experiment.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    train()
