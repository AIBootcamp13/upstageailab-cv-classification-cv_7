import pandas as pd
import hydra
import os
import sys
import torch
from collections import defaultdict
from pytorch_lightning import Trainer
from hydra.utils import instantiate

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset.datamodule_kfold import DocumentDataModule

@hydra.main(config_path="../../configs", config_name="config")
def test(cfg):
    img_logits_dict = defaultdict(list)

    for fold in range(cfg.data.num_folds):
        print(f"🔍 Predicting with fold {fold}...")

        # DataModule 설정
        dm = DocumentDataModule(**{**cfg.data, "fold": fold})
        dm.setup("predict")

        # 모델 로딩
        model_path = os.path.join(ROOT_DIR, "artifacts", cfg.model.name, f"{cfg.model.name}_fold{fold}.pt")
        model = instantiate(cfg.model)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        # 추론
        trainer = Trainer(**cfg.train)
        predictions = trainer.predict(model, datamodule=dm)

        for batch_output in predictions:
            img_names = batch_output["img_name"]
            logits = batch_output["logits"].cpu()

            for name, logit in zip(img_names, logits):
                img_logits_dict[name].append(logit)

    # soft voting → 평균 logits 후 argmax
    result = []
    for img_name, logit_list in img_logits_dict.items():
        avg_logit = torch.stack(logit_list).mean(dim=0)
        pred_class = torch.argmax(avg_logit).item()
        result.append({"ID": img_name, "target": pred_class})

    # DataFrame 정리 및 저장
    result_df = pd.DataFrame(result).sort_values("ID")
    result_df.to_csv(os.path.join(ROOT_DIR, f"{cfg.model.name}_kfold.csv"), index=False)
    print("✅ Soft voting complete. Submission saved.")

if __name__ == "__main__":
    test()
