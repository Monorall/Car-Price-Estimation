import torch
import pandas as pd
from utils import prepare_dataset
from torch.utils.data import Dataset, DataLoader


class CarsDataset(Dataset):
    def __init__(self, df):
        features = df.drop(columns=["price"])
        targets = df["price"]

        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        feature = self.features[index]
        target = self.targets[index]
        return feature, target


if __name__ == "__main__":
    dframe = pd.read_csv("data/cars_dataset.csv")
    dframe = prepare_dataset(dframe)
    dataset = CarsDataset(dframe)
    print(0)
