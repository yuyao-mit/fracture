import os
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class FractureDataset(Dataset):
    """
    Single-GPU Dataset for fracture data
    Supports multi-step rollout targets: y(t+1), y(t+2), y(t+3)
    """

    def __init__(self, folder, rollout_steps=3):
        self.samples = []
        self.rollout_steps = rollout_steps

        self.files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npz")
        ])

        for path in self.files:
            data = np.load(path)
            inputs = data["inputs"]
            targets = data["targets"]

            n = inputs.shape[0]

            # 保证 i + rollout_steps 不越界
            for i in range(n - rollout_steps):
                self.samples.append((path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            path, i = self.samples[idx]
            data = np.load(path)

            # x(t)
            x = torch.from_numpy(data["inputs"][i]).float()     # [C, H, W]

            # y(t+1 ... t+K)
            y = torch.from_numpy(
                data["targets"][i + 1 : i + 1 + self.rollout_steps]
            ).float()                                           # [K, 1, H, W]

            if torch.isnan(x).any() or torch.isnan(y).any():
                raise ValueError("NaN detected")

            return {
                "image": x,     # x(t)
                "mask": y       # [y(t+1), y(t+2), y(t+3)]
            }

        except Exception as e:
            print(
                f"[Dataset Error] File: {path}, Index: {i}, Error: {e}"
            )
            raise
