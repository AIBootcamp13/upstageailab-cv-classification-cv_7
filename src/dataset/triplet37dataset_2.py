import random
from torch.utils.data import Dataset

class BalancedClass3and7Dataset(Dataset):
    def __init__(self, base_dataset, target_classes=[3, 7]):
        self.class_to_samples = {cls: [] for cls in target_classes}

        for img, label in base_dataset:
            if label in target_classes:
                self.class_to_samples[label].append((img, label))

        # 가장 적은 클래스 기준으로 길이 맞춤
        self.length = min(len(self.class_to_samples[3]), len(self.class_to_samples[7])) * 2
        self.classes = target_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 홀수 idx → class 3, 짝수 idx → class 7 (또는 반대로)
        cls = self.classes[idx % 2]
        sample_list = self.class_to_samples[cls]
        return random.choice(sample_list)