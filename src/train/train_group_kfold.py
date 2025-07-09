import hydra
import os
import sys
import torch
import wandb
import time
import pandas as pd

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


# 현재 파일 기준으로 프로젝트 루트 경로 찾기 (group-kfold.py와 동일)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
print(ROOT_DIR)

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from hydra.utils import instantiate
from src.dataset.datamodule_group_kfold import DocumentDomainAugDataModule  # 새로운 DataModule
import importlib.util

@hydra.main(config_path="../../configs", config_name="config")
def train(cfg):
    wandb.login()

    model_path = os.path.join(ROOT_DIR, "artifacts")

    # train_group_kfold.csv 파일 존재 확인 (없으면 에러)
    group_kfold_path = f"{ROOT_DIR}/data/train_group_kfold.csv"
    if not os.path.exists(group_kfold_path):
        raise FileNotFoundError(
            f"GroupKFold 파일이 없습니다: {group_kfold_path}\n"
            f"먼저 group-kfold.py를 실행하여 파일을 생성하세요:\n"
            f"python src/utils/group-kfold.py"
        )

    # 모델 저장 폴더 생성
    if not os.path.exists(os.path.join(ROOT_DIR, "artifacts")):
        os.makedirs(os.path.join(ROOT_DIR, "artifacts"))

    for fold in range(cfg.data.num_folds):
        start_time = time.time()

        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.wandb.run_name}_fold{fold}",
            group=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True
        )

        early_stop_callback = EarlyStopping(
            monitor="val/f1",   # 기준 지표
            mode="max",          
            patience=3,          # 성능이 개선되지 않는 epoch 수
            verbose=True
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val/f1",
            mode="max",
            save_top_k=1,
            save_last=False,
            dirpath=model_path,
            filename=f"{cfg.model.name}_fold{fold}"
        )
        
        try:
            # 새로운 DocumentDomainAugDataModule 사용
            dm = DocumentDomainAugDataModule(**{**cfg.data, "fold": fold})
            model = instantiate(cfg.model)
            trainer = Trainer(
                **cfg.train, 
                callbacks=[early_stop_callback, checkpoint_callback], 
                logger=wandb_logger
            )
            
            print(f"\n=== Fold {fold} 학습 시작 ===")
            print(f"Data config: {cfg.data}")
            
            trainer.fit(model, datamodule=dm)

            best_model_path = checkpoint_callback.best_model_path
            best_model = instantiate(cfg.model)
            best_model.load_state_dict(torch.load(best_model_path)['state_dict'])

            # 학습 시간 및 best score 기록
            end_time = time.time()
            total_time_sec = end_time - start_time
            total_time_min = total_time_sec / 60
            best_f1 = checkpoint_callback.best_model_score
            
            print(f"Fold {fold} 완료!")
            print(f"Training time: {total_time_min:.2f} minutes")
            print(f"Best F1 score: {best_f1:.4f}")

            # WandB 로깅
            wandb_logger.experiment.log({
                "training_time_sec": total_time_sec, 
                "training_time_min": total_time_min, 
                "best_f1": best_f1,
                "fold": fold
            })
            
            # 모델 저장
            pt_path = os.path.join(ROOT_DIR, "artifacts", f"{cfg.model.name}_fold{fold}.pt")
            torch.save(best_model.state_dict(), pt_path)

            # WandB Artifact 저장
            artifact = wandb.Artifact(name=f"{cfg.model.name}_fold{fold}", type="model")
            artifact.add_file(pt_path)
            wandb_logger.experiment.log_artifact(artifact)

        except Exception as e:
            print(f"Fold {fold} 학습 중 에러 발생: {e}")
            raise e
        finally:
            wandb_logger.experiment.finish()
    
    print("\n=== 전체 학습 완료 ===")
    wandb.finish()

if __name__ == "__main__":
    train()