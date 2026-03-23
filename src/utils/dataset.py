import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, folders, train_mode=True):
        self.samples = []
        self.train_mode = train_mode

        if isinstance(folders, str):
            folders = [folders]

        for folder in folders:
            paths = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(".npz")
            ])

            for path in paths:
                data = np.load(path, mmap_mode="r")
                n = data["inputs"].shape[0]

                for i in range(n):
                    self.samples.append((path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, i = self.samples[idx]

        data = np.load(path, mmap_mode="r")

        x = torch.from_numpy(data["inputs"][i]).float()
        y = torch.from_numpy(data["targets"][i]).float()
        lc = torch.tensor(data["lc"][i]).float()

        return {
            "image": x,
            "mask": y,
            "lc": lc,
            "train_mode": self.train_mode
        }