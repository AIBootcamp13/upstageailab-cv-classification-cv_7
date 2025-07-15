import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU일 때

    # CuDNN 결정적 연산 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 환경 변수로 Python 해시 랜덤시드 고정
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Global seed set to {seed}")
    

#사용 시,    
#from utils.seed import set_seed
# Hydra config 불러온 다음에
#set_seed(config.train.seed)  # 예: config.train.seed = 42

