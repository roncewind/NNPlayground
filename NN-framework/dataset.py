# dataset.py
import numpy as np
import pandas as pd
import torch
from config import NUM_CLASSES, NUM_INPUTS
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_path, feature_extractor, label_col="label"):
        self.data = pd.read_csv(csv_path)
        self.label_col = label_col
        self.feature_extractor = feature_extractor
        self.labels = self.data[label_col].values
        self.features = [self.feature_extractor(row) for _, row in self.data.iterrows()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# -----------------------------------------------------------------------------
def feature_extractor(row):
    # Replace with real logic later. Assumes all features are in columns except 'label'
    x = dummy_feature_extractor(row)
    return x


# -----------------------------------------------------------------------------
def dummy_feature_extractor(row):
    # Replace with real logic later. Assumes all features are in columns except 'label'
    vec = np.array(row.drop("label")).astype(np.float32)
    if len(vec) < NUM_INPUTS:
        vec = np.pad(vec, (0, NUM_INPUTS - len(vec)))
    return vec[:NUM_INPUTS]


# -----------------------------------------------------------------------------
def generate_random_csv(path="random_data.csv", num_samples=1000, seed=42):
    np.random.seed(seed)
    features = np.random.rand(num_samples, NUM_INPUTS)
    labels = np.random.randint(0, NUM_CLASSES, size=(num_samples, 1))

    columns = [f"f{i}" for i in range(NUM_INPUTS)] + ["label"]
    data = np.hstack([features, labels])
    df = pd.DataFrame(data, columns=columns)
    df["label"] = df["label"].astype(int)

    df.to_csv(path, index=False)
    print(f"✅ Random dataset saved to: {path}")
