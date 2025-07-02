from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms

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

        image = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])(image)

        return image, img_name  # label 없음
