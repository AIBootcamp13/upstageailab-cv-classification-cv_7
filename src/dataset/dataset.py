from PIL import Image
from torch.utils.data import Dataset
import os
import random
import numpy as np

class DocumentDataset(Dataset):
    def __init__(self, df_subset, data_dir, apply_transform_prob = 1.0, aug_pipeline=None, transform=None):
        self.df = df_subset
        self.data_dir = data_dir
        self.transform = transform
        self.apply_transform_prob = apply_transform_prob
        self.aug_pipeline = aug_pipeline

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, label = self.df[idx]
        img_path = os.path.join(self.data_dir, "train", img_name)

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.aug_pipeline:
            image = self.aug_pipeline(image)

        # # 확률적 augraphy 증강 적용
        # prob = random.random()
        # if self.aug_pipeline and (prob >= self.apply_transform_prob):
        #     image = self.aug_pipeline(image)

        if self.transform:
            image = self.transform(image=image)
            image = image['image'] #(C,H,W) tensor 포맷
        

        return image, label
