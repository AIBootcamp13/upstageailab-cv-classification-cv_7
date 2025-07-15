import torch
import random
import numpy as np
from torch.utils.data import Dataset

class PureCutMixDataset(Dataset):
    def __init__(self, base_dataset, cutmix_classes=[3, 4, 7, 14], alpha=1.0):
        """
        base_dataset: (image, label)을 반환하는 Dataset
        cutmix_classes: CutMix를 적용할 클래스 목록
        alpha: beta 분포의 alpha 값
        """
        self.base_dataset = base_dataset
        self.alpha = alpha
        self.cutmix_classes = set(cutmix_classes)

        # 해당 클래스들만 필터링
        self.filtered_indices = [
            i for i in range(len(base_dataset)) 
            if base_dataset[i][1] in self.cutmix_classes
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, i):
        idx1 = self.filtered_indices[i]
        x1, y1 = self.base_dataset[idx1]

        # 두 번째 이미지도 같은 클래스 제한 없이 랜덤
        idx2 = random.choice(self.filtered_indices)
        x2, y2 = self.base_dataset[idx2]

        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x1.size(), lam)

        # CutMix 적용
        x1 = x1.clone()  # tensor 수정 방지
        x1[:, bby1:bby2, bbx1:bbx2] = x2[:, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1.size(-1) * x1.size(-2)))

        return x1, y1, y2, lam

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
