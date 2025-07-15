import os
import sys
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print(ROOT_DIR)

def load_dataframe():
    return pd.read_csv(f"{ROOT_DIR}/data/train.csv")

def get_feature_extractor():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return model, transform

def extract_features(train_df, model, transform):
    features = []
    for img_id in train_df['ID']:  # id 컬럼명 주의!
        img_path = os.path.join(ROOT_DIR, "data", "train",img_id)
        features.append(get_feature(img_path, model, transform))
    return np.stack(features)

def get_feature(img_path, model, transform):
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feature = model(tensor).squeeze().numpy()
    return feature

def assign_clusters(features, n_clusters=50):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)

def assign_groupkfold(train_df, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    train_df["kfold"] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(X=train_df, y=train_df["target"], groups=train_df["group"])):
        train_df.loc[val_idx, "kfold"] = fold
    return train_df

def save_df(train_df):
    train_df.to_csv(f"{ROOT_DIR}/data/train_group_kfold.csv", index=False)
    print("train_fold.csv 저장 완료!")

if __name__ == "__main__":
    train_df = load_dataframe()
    model, transform = get_feature_extractor()
    features = extract_features(train_df, model, transform)
    clusters = assign_clusters(features, n_clusters=50)
    train_df['group'] = clusters
    train_df = assign_groupkfold(train_df, n_splits=5)
    save_df(train_df)
