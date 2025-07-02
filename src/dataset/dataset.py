from PIL import Image
from torch.utils.data import Dataset
import os
import random
import numpy as np
from torchvision import transforms

class DocumentDataset(Dataset):
    def __init__(self, df_subset, data_dir, apply_transform_prob = 1.0, transform=None):
        self.df = df_subset
        self.data_dir = data_dir
        self.transform = transform
        self.apply_transform_prob = apply_transform_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, label = self.df[idx]
        img_path = os.path.join(self.data_dir, "train", img_name)
        image = Image.open(img_path).convert("RGB")

        # if self.transform:
        #     image = self.transform(image)

        #확률적 증강 적용
        if self.transform and random.random() < self.apply_transform_prob:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(image)

        return image, label
