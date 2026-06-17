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


from src.dataset_cached import (
    CachedECGDataset
)

dataset = CachedECGDataset(
    "../data/processed"
)



loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
)


start = time.time()

samples_seen = 0

for batch_idx, (x, y) in enumerate(loader):

    samples_seen += len(x)

    if batch_idx >= 100:
        break

elapsed = time.time() - start

print("\n=== Cached Dataset ===")
print(f"Batches processed : {batch_idx+1}")
print(f"Samples processed : {samples_seen:,}")
print(f"Elapsed time      : {elapsed:.2f} s")
print(
    f"Throughput        : "
    f"{samples_seen/elapsed:.1f} samples/s"
)


