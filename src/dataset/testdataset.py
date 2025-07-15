from PIL import Image
from torch.utils.data import Dataset
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        test_dir = os.path.join(self.data_dir, "test")
        self.image_list = sorted([
            f for f in os.listdir(test_dir)
            if os.path.isfile(os.path.join(test_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.data_dir, "test", img_name)
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            for transform in self.transform:
                image = transform(image=image)
                image = image['image'] #(C,H,W) tensor 포맷

        return image, img_name  # label 없음
