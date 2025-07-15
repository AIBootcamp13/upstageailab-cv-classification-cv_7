import pandas as pd
import hydra
import os
import sys
import torch
import wandb

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pytorch_lightning import Trainer
from hydra.utils import instantiate
from src.dataset.datamodule import DocumentDataModule

@hydra.main(config_path="../../configs", config_name="config")
def test(cfg):
    # wandb에서 모델 파라미터 불러오기
    # wandb.login()
    # run = wandb.init(project="문서분류")
    # artifact = run.use_artifact("your-username/resnet-model:v0", type="model")
    # artifact_dir = artifact.download()

    dm = DocumentDataModule(**cfg.data)
    dm.setup("predict")

    model_path = os.path.join(ROOT_DIR, "artifacts", f"{cfg.model.name}", f"{cfg.model.name}.pt")
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(model_path))

    trainer = Trainer(**cfg.train)
    predictions = trainer.predict(model, datamodule=dm)

    all_img_names = []
    all_preds = []

    for batch_output in predictions:
        all_img_names.extend(batch_output["img_name"])
        all_preds.extend(batch_output["pred"].cpu().numpy())
    
    pred_df = pd.DataFrame({"ID": all_img_names, "target": all_preds})
    pred_df = pred_df.sort_values(by="ID")
    pred_df.to_csv(os.path.join(ROOT_DIR, f"{cfg.model.name}.csv"), index=False)

    # wandb.finish()

if __name__ == "__main__":
    test()