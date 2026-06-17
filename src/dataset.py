from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class CachedECGDataset(Dataset):

    def __init__(self, data_dir):

        self.X = []
        self.y = []

        files = sorted(
            Path(data_dir).glob("*.npz")
        )

        for file in files:

            data = np.load(file)

            self.X.append(data["X"])
            self.y.append(data["y"])

        self.X = np.concatenate(self.X)
        self.y = np.concatenate(self.y)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        x = torch.tensor(
            self.X[idx],
            dtype=torch.float32,
        )

        y = torch.tensor(
            self.y[idx],
            dtype=torch.long,
        )

        return x, y



class ECGDataset(Dataset):

    def __init__(self, data_dir):

        self.samples = []

        files = sorted(Path(data_dir).glob("*.npz"))
        print(f"files read {files}")
        for file in files:

            data = np.load(file)

            n_windows = len(data["y"])

            for idx in range(n_windows):

                self.samples.append(
                    (file, idx)
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        file, window_idx = self.samples[idx]

        data = np.load(file)

        x = data["X"][window_idx]
        y = data["y"][window_idx]

        x = torch.tensor(
            x,
            dtype=torch.float32,
        )

        y = torch.tensor(
            y,
            dtype=torch.long,
        )

        return x, y


class FastCachedECGDataset(Dataset):

    def __init__(self, data_dir):

        X = []
        y = []

        files = sorted(
            Path(data_dir).glob("*.npz")
        )

        for file in files:

            data = np.load(file)

            X.append(data["X"])
            y.append(data["y"])

        self.X = torch.tensor(
            np.concatenate(X),
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            np.concatenate(y),
            dtype=torch.long,
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx],
        )
