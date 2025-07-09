from hydra.utils import instantiate
import pandas as pd
import hydra
import os
import sys
import torch

from collections import defaultdict
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.dataset.datamodule_triplet import DocumentDataModule
from torch.utils.data import DataLoader
from src.utils.pil import pil_to_base64, tensor_to_thumbnail_base64

@hydra.main(config_path="../../configs", config_name="config")
def main(cfg):
  model_name = cfg.model.name
  pred_list = defaultdict(list)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")



  # 데이터셋 준비
  dm = DocumentDataModule(**cfg.data)
  dm.setup("analyze")
  analyze_dataset = dm.analyze_dataset
  if analyze_dataset is None:
    raise ValueError("analyze_dataset is not initialized")

  # 모델 준비
  model_path = os.path.join(ROOT_DIR, 'artifacts', model_name, f'{model_name}.pt')
  model = instantiate(cfg.model)
  model.load_state_dict(torch.load(model_path))
  model.to(device)
  model.eval()
  
  # 데이터로더 준비
  analyze_loader = DataLoader(analyze_dataset, batch_size=32, shuffle=False)
  with torch.no_grad():
      for batch in tqdm(analyze_loader):
        output = model.predict_step_analyze(batch)
        pred_list["img_name"].extend(output["img_name"])
        pred_list["pred"].extend(output["pred"])
        pred_list["label"].extend(output["label"])
        img_base64 = [
          tensor_to_thumbnail_base64(img_tensor, size=(192, 192))
          for img_tensor in output["img"]
        ]
        pred_list["img"].extend(img_base64)
  

  df = pd.DataFrame(pred_list)
  df.to_csv(os.path.join(ROOT_DIR, "data", "misclassification_triplet.csv"), index=False)
  

if __name__ == "__main__":
  main()