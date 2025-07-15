import random
from torch.utils.data import Dataset

class Class3and7TripletDataset(Dataset):
    def __init__(self, base_dataset, target_classes=[3, 7]):
        """
        Args:
            base_dataset: (img, label) 반환하는 기본 Dataset
            target_classes: Triplet 학습 대상으로 쓸 클래스들 (여기선 3, 7)
        """
        self.base_dataset = base_dataset
        self.target_classes = target_classes

        self.targets = [label for _, label in base_dataset]
        self.class_to_indices = {cls: [] for cls in target_classes}
        for idx, label in enumerate(self.targets):
            if label in target_classes:
                self.class_to_indices[label].append(idx)

        self.available_indices = sum(self.class_to_indices.values(), [])  # anchor용 인덱스

    def __len__(self):
        return len(self.available_indices)

    def __getitem__(self, _):
        # Anchor 선택
        anchor_idx = random.choice(self.available_indices)
        anchor_img, anchor_label = self.base_dataset[anchor_idx]

        # Positive 선택 (같은 클래스)
        pos_idx = anchor_idx
        while pos_idx == anchor_idx:
            pos_idx = random.choice(self.class_to_indices[anchor_label])
        positive_img, _ = self.base_dataset[pos_idx]

        # Negative 선택 (다른 클래스)
        negative_label = [c for c in self.target_classes if c != anchor_label][0]
        negative_idx = random.choice(self.class_to_indices[negative_label])
        negative_img, _ = self.base_dataset[negative_idx]

        return anchor_img, positive_img, negative_img
