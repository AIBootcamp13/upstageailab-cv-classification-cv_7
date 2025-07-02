import hydra
from omegaconf import DictConfig
import os
import torch
from src.datamodule import DocumentDataModule  # 모듈 이름에 맞게 import 경로 수정

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 데이터 모듈 초기화
    dm = DocumentDataModule(**cfg.data)

    # 데이터셋 준비
    dm.setup()

    # 학습용 배치 확인
    print("==== Training Batch ====")
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        x, y = batch  # x: 이미지 tensor, y: 레이블
        print("Train batch image shape:", x.shape)
        print("Train batch labels:", y)
        break  # 한 배치만 확인

    # 검증용 배치 확인
    print("==== Validation Batch ====")
    val_loader = dm.val_dataloader()
    for batch in val_loader:
        x, y = batch
        print("Val batch image shape:", x.shape)
        print("Val batch labels:", y)
        break

if __name__ == "__main__":
    main()
