from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from augraphy import AugraphyPipeline, InkBleed, Brightness, NoiseTexturize
import pandas as pd
import os
import hydra
import sys
import numpy as np
from PIL import Image
# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
from omegaconf import DictConfig
from src.dataset.dataset import DocumentDataset
from src.dataset.testdataset import TestDataset

class DocumentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=32, num_workers=4, val_split=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.df = pd.read_csv(os.path.join(self.data_dir, "train.csv")).values
        # AugraphyPipeline 정의
        aug_pipeline = AugraphyPipeline(
            ink_phase=[InkBleed(
                intensity_range=(0.4, 0.7),
                kernel_size=(5, 5),
                severity=(0.2, 0.4)
            )],
            paper_phase=[],
            post_phase=[
                Brightness(
                    brightness_range=(0.7, 1.3),
                    min_brightness=0,
                ),
                NoiseTexturize(
                    sigma_range=(2, 3),
                    turbulence_range=(2, 5),
                    texture_width_range=(50, 500),
                    texture_height_range=(50, 500),
                )
            ]
        )

        # 최종 transform
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: np.array(x)),
            aug_pipeline,
            transforms.Lambda(lambda x: Image.fromarray(x)),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        train_df, val_df = train_test_split(
            self.df,
            test_size=self.val_split,
            stratify=self.df[:, 1],  # 라벨 기준 stratify
            random_state=42
        )
        self.train_dataset = DocumentDataset(train_df, self.data_dir, apply_transform_prob=0.5, transform=self.transform)
        self.val_dataset = DocumentDataset(val_df, self.data_dir)
        self.test_dataset = TestDataset(self.data_dir)
        # full_dataset = ImageFolder(os.path.join(self.data_dir, "train"))
        # train_label_csv = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        # targets = full_dataset.targets  # 클래스 인덱스 리스트
        # indices = np.arange(len(full_dataset))

        # train_idx, val_idx = train_test_split(
        #     indices,
        #     test_size=self.val_split,
        #     stratify=targets,
        #     random_state=42
        # )

        # self.train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        # self.val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

        # # transform은 Subset 내부의 dataset에 직접 설정
        # full_dataset.transform = self.transform

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset is not initialized")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("val_dataset is not initialized")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("test_dataset is not initialized")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    dm = DocumentDataModule(**cfg.data)
    dm.setup()

if __name__ == "__main__":
    main()