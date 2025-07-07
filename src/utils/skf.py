# utils/kfold_split.py
import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def make_stratified_kfold(df: pd.DataFrame, n_splits=5, seed=42):
    df = df.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["kfold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(X=df, y=df["target"])):
        df.loc[val_idx, "kfold"] = fold

    return df

if __name__ == "__main__":
    df = make_stratified_kfold(pd.read_csv(f"{ROOT_DIR}/data/train.csv"), n_splits=5, seed=42)
    df.to_csv(f"{ROOT_DIR}/data/train_fold.csv", index=False)