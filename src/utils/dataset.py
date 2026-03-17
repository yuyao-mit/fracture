import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, folder):
        """
        Args:
            folder: directory containing .npz files
                    e.g. data/train , data/val , data/test
        """
        self.samples = []

        # collect all npz files
        paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npz")
        ])

        # build sample index
        for path in paths:
            data = np.load(path, mmap_mode="r")  # faster for large files
            n = data["inputs"].shape[0]

            for i in range(n):
                self.samples.append((path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, i = self.samples[idx]

        data = np.load(path, mmap_mode="r")

        x = torch.from_numpy(data["inputs"][i]).float()   # [C,H,W]
        y = torch.from_numpy(data["targets"][i]).float()  # [1,H,W]
        lc = torch.tensor(data["lc"][i]).float()          # scalar

        return {
            "image": x,
            "mask": y,
            "lc": lc
        }