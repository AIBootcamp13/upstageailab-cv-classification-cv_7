from PIL import Image
from torch.utils.data import Dataset
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_list = os.listdir(os.path.join(self.data_dir, "test"))
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.data_dir, "test", img_name)
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        image = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])(image=image)['image']

        return image, img_name  # label 없음
