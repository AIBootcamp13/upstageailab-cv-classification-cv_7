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
    
print(ROOT_DIR)

from src.dataset.datamodule_group_kfold import DocumentDomainAugDataModule

@hydra.main(config_path="../../configs", config_name="config")
def test(cfg):
    """Group KFold로 학습된 모델로 앙상블 추론"""
    
    img_logits_dict = defaultdict(list)
    
    for fold in range(cfg.data.num_folds):
        print(f"🔍Predicting with fold {fold} ...")
        
        try:
            dm = DocumentDomainAugDataModule(**{**cfg.data, "fold": fold})
            dm.setup("predict") #test_dataset만 생성
        except FileNotFoundError as e:
            print(f"❌필요한 파일이 없습니다: {e}")
            print("group_kfold.py를 실행이 되었는지 확인이 필요헙니다.")
            sys.exit(1)
            
        model_path = os.path.join(ROOT_DIR, "artifacts", f"{cfg.model.name}_fold{fold}.pt")
        
        try: #저장된 모델 가중치 로드
            model = instantiate(cfg.model)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            
        except FileNotFoundError:
            print(f"❌모델 파일이 없습니다.: {model_path}")
            print("group_kfold.py를 실행이 되었는지 확인이 필요헙니다.")
            sys.exit(1)
            
        #추론 실행    
        trainer = Trainer(**cfg.train)
        predictions = trainer.predict(model, datamodule=dm)
        
        #배치별 결과 수집
        for batch_output in predictions:
            img_names = batch_output["img_name"]
            logits = batch_output["logits"].cpu()
            
            for name, logit in zip(img_names, logits):
                img_logits_dict[name].append(logit)
                
    print(f"✅ 모든 Fold 추론 완료: 총 {len(img_logits_dict)}개 이미지")
    
    # Soft Voting -> 평균 logits 후 argmax
    result = []
    for img_name, logit_list in img_logits_dict.items():
        avg_logit = torch.stack(logit_list).mean(dim=0)
        pred_class = torch.argmax(avg_logit).item()
        result.append({"ID": img_name, "target": pred_class})
        
    result_df = pd.DataFrame(result).sort_values("ID")
    output_path = os.path.join(ROOT_DIR, f"{cfg.model.name}_group_kfold.csv")
    result_df.to_csv(output_path, index=False)
    
    print(f"✅ Soft Voting complete. Submission saved to {output_path}")
    print(f"📊 결과 확인 : {len(result_df)}개 샘플 예측 완료")
    
    #클래스 분포 확인
    class_distribution = result_df["target"].value_counts().sort_index()
    print("📈 예측 클래스 분포 :")
    for class_id, count in class_distribution.items():
        print(f"Class{class_id}: {count}개")
            
            
if __name__ == "__main__":
    test()