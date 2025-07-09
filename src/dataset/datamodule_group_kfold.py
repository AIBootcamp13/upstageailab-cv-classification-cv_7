import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from collections import Counter
import hydra
import sys
from omegaconf import DictConfig



ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
print(ROOT_DIR)

from src.dataset.dataset import DocumentDataset
from src.dataset.testdataset import TestDataset
from src.transform.custom_transform_group import (
    get_common_transfom,
    get_compression_pipeline,
    get_skew_pipeline,
    get_inkbleed_pipeline,
    get_brightness_pipeline,  # 새로 추가
    get_test_transform,
)

class DocumentDomainAugDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir="data", 
                 batch_size=32, 
                 num_workers=4, 
                 fold=0,
                 num_folds=5,
                 fold_path=None,
                 image_size=(224, 224),
                 augmentation_intensity="medium",
                 balance_classes=True,
                 use_val_augmentation=False,
                 **kwargs):  # 추가 파라미터 받기
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        self.num_folds = num_folds
        
        if fold_path is None:
            self.fold_path = f"{ROOT_DIR}/data/train_group_kfold.csv"
        else:
            # config 경로가 상대경로면 절대경로로 변경
            if fold_path == "data/train_group_kfold.csv":
                self.fold_path = f"{ROOT_DIR}/data/train_group_kfold.csv"
            else:
                self.fold_path = fold_path
            
        self.image_size = image_size
        self.augmentation_intensity = augmentation_intensity
        self.balance_classes = balance_classes
        self.use_val_augmentation = use_val_augmentation
        
        if not os.path.exists(self.fold_path):
            raise FileNotFoundError(f"Fold 파일이 없습니다: {self.fold_path}")
        
        print(f"Fold 파일 로드: {self.fold_path}")
        self.df = pd.read_csv(self.fold_path)
        self._validate_dataframe()

    def _validate_dataframe(self):
        """데이터프레임 검증"""
        required_columns = ["ID", "target", "kfold"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        if self.fold not in self.df["kfold"].unique():
            raise ValueError(f"Fold {self.fold}가 데이터에 없습니다. 사용 가능한 fold: {self.df['kfold'].unique()}")
        
        print(f"데이터 로드 완료: {len(self.df)}개 샘플, Fold {self.fold} 사용")

    def _get_domain_mapping(self):
        """새로운 도메인별 클래스 매핑 (brightness 추가)"""
        return {
            # 의료: 잉크 번짐 중심 (스탬프, 서명, 수기)
            "medical": {
                "targets": [1, 4, 6, 7, 11, 12, 14],  # 의료 관련 7개
                "primary_aug": "inkbleed",
                "secondary_aug": ["skew"]
            },
            
            # 신분증: 회전/기울기 중심 (촬영 각도)
            "id_docs": {
                "targets": [5, 8, 9],  # 운전면허증, 주민등록증, 여권
                "primary_aug": "skew",
                "secondary_aug": ["compression"]
            },
            
            # 차량: 밝기 조정 중심 (EDA 인사이트 반영!)
            "vehicle": {
                "targets": [2, 15, 16],  # 대시보드, 등록증, 번호판
                "primary_aug": "brightness",  # 새로 변경!
                "secondary_aug": ["compression"]  # 압축도 함께
            },
            
            # 금융: 압축 중심 (깔끔한 텍스트)
            "financial": {
                "targets": [0, 10],  # 계좌번호, 결제확인서
                "primary_aug": "compression",
                "secondary_aug": []
            },
            
            # 일반: 표준 처리
            "general": {
                "targets": [3, 13],  # 입퇴원확인서, 이력서
                "primary_aug": "compression",
                "secondary_aug": ["skew"]
            }
        }

    def _get_augmentation_pipeline(self, aug_type):
        """증강 파이프라인 반환 (brightness 추가)"""
        pipelines = {
            "compression": get_compression_pipeline(),
            "skew": get_skew_pipeline(),
            "inkbleed": get_inkbleed_pipeline(),
            "brightness": get_brightness_pipeline(),  # 새로 추가
        }
        
        return pipelines.get(aug_type)

    def _balance_dataset(self, df):
        """클래스 불균형 해결"""
        if not self.balance_classes:
            return df
        
        class_counts = df["target"].value_counts()
        print(f"클래스 분포: {dict(class_counts)}")
        
        median_count = class_counts.median()
        balanced_dfs = []
        
        for target_class in class_counts.index:
            class_df = df[df["target"] == target_class]
            current_count = len(class_df)
            
            if current_count < median_count:
                repeat_factor = int(median_count / current_count)
                balanced_dfs.extend([class_df] * repeat_factor)
                
                remaining = int(median_count % current_count)
                if remaining > 0:
                    balanced_dfs.append(class_df.sample(remaining, replace=True))
            else:
                balanced_dfs.append(class_df)
        
        result_df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"밸런싱 후 분포: {dict(result_df['target'].value_counts())}")
        print(f"반환값 타입 : {type(result_df)}")
        return result_df

    def setup(self, stage=None):
        """데이터셋 설정"""
        df = self.df
        train_df = df[df["kfold"] != self.fold][["ID", "target"]].reset_index(drop=True)
        val_df = df[df["kfold"] == self.fold][["ID", "target"]].reset_index(drop=True)
        
        print(f"Train: {len(train_df)}개, Val: {len(val_df)}개")
        print(f"Train Type: {type(train_df)}, Val Type: {type(val_df)}")
        train_df = self._balance_dataset(train_df)
        
        transform_common = get_common_transfom(img_size=self.image_size)
        transform_test = get_test_transform(img_size=self.image_size)
        
        # === Train Dataset 구성 ===
        
        if self.data_dir == "data":
            absolute_data_dir = f"{ROOT_DIR}/data"
        else: 
            absolute_data_dir = os.path.abspath(self.data_dir)
            
        print(f"데이터 디렉토리 : {absolute_data_dir}")
        
        train_datasets = []
        domain_mapping = self._get_domain_mapping()
        
        # # data_dir 절대경로 변환 (train 폴더까지 포함)
        # if self.data_dir == "data":
        #     absolute_train_dir = f"{ROOT_DIR}/data/train"
        # else:
        #     absolute_train_dir = os.path.join(self.data_dir, "train")
            
        # 1. 원본 데이터셋 (증강 없음)
        train_datasets.append(
            DocumentDataset(train_df.values, 
                          absolute_data_dir, #os.path.join(self.data_dir)에서 절대경로로 변경
                          aug_pipeline=None, 
                          transform=transform_test)
        )
        
        # 2. 공통 증강 데이터셋 (Motion Blur 포함)
        train_datasets.append(
            DocumentDataset(train_df.values, 
                          absolute_data_dir, #os.path.join(self.data_dir)에서 절대경로로 변경
                          aug_pipeline=None, 
                          transform=transform_common)
        )
        
        # 3. 도메인별 특화 증강 (3개 파이프라인)
        for domain_name, domain_info in domain_mapping.items():
            targets = domain_info["targets"]
            primary_aug = domain_info["primary_aug"]
            secondary_augs = domain_info.get("secondary_aug", [])
            
            domain_df = train_df[train_df["target"].isin(targets)]
            if len(domain_df) == 0:
                continue
            
            print(f"{domain_name} 도메인: {len(domain_df)}개 샘플")
            
            # Primary 증강
            primary_pipeline = self._get_augmentation_pipeline(primary_aug)
            if primary_pipeline:
                train_datasets.append(
                    DocumentDataset(domain_df.values,
                                  absolute_data_dir, #os.path.join(self.data_dir)에서 절대경로로 변경
                                  aug_pipeline=primary_pipeline,
                                  transform=transform_common)
                )
            
            # Secondary 증강 (절반만 적용)
            for secondary_aug in secondary_augs:
                secondary_pipeline = self._get_augmentation_pipeline(secondary_aug)
                if secondary_pipeline:
                    subset_df = domain_df.sample(frac=0.5, random_state=42)
                    train_datasets.append(
                        DocumentDataset(subset_df.values,
                                      absolute_data_dir, #os.path.join(self.data_dir)에서 절대경로로 변경
                                      aug_pipeline=secondary_pipeline,
                                      transform=transform_common)
                    )
        
        self.train_dataset = ConcatDataset(train_datasets)
        
        # === Validation Dataset 구성 (3배 확장) ===
        val_datasets = []
        
        # 1. 원본 (증강 없음)
        val_datasets.append(
            DocumentDataset(
                val_df.values, 
                absolute_data_dir, #os.path.join(self.data_dir)에서 절대경로로 변경
                aug_pipeline=None, 
                transform=transform_test
            )
        )
        
        # 2. 공통 증강
        val_datasets.append(
            DocumentDataset(
                val_df.values, 
                absolute_data_dir, #os.path.join(self.data_dir)에서 절대경로로 변경
                aug_pipeline=None, 
                transform=transform_common
            )
        )
        
        # 3. 도메인별 대표 증강 (각 도메인의 primary 증강만)
        for domain_name, domain_info in domain_mapping.items():
            targets = domain_info["targets"]
            primary_aug = domain_info["primary_aug"]
            
            domain_val_df = val_df[val_df["target"].isin(targets)]
            if len(domain_val_df) == 0:
                continue
                
            primary_pipeline = self._get_augmentation_pipeline(primary_aug)
            if primary_pipeline:
                val_datasets.append(
                    DocumentDataset(
                        domain_val_df.values,
                        absolute_data_dir, #os.path.join(self.data_dir)에서 절대경로로 변경
                        aug_pipeline=primary_pipeline,
                        transform=transform_common
                    )
                )
        
        self.val_dataset = ConcatDataset(val_datasets)
        
        # === Test Dataset ===
        # data_dir도 절대경로로 수정
        if self.data_dir == "data":
            absolute_data_dir = f"{ROOT_DIR}/data"
        else:
            absolute_data_dir = self.data_dir
            
        self.test_dataset = TestDataset(absolute_data_dir, transform=transform_test)
        
        print(f"\n=== 최종 데이터셋 크기 ===")
        print(f"Train: {len(self.train_dataset):,}개")
        print(f"Val: {len(self.val_dataset):,}개") 
        print(f"Test: {len(self.test_dataset):,}개")
        print(f"증강 배율: {len(self.train_dataset) / len(train_df):.1f}배")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Hydra를 사용한 메인 함수"""
    dm = DocumentDomainAugDataModule(**cfg.data)
    dm.setup()
    
    train_loader = dm.train_dataloader()
    print(f"Train batches: {len(train_loader)}")
    print("✅ DataModule 설정 완료!")

if __name__ == "__main__":
    main()