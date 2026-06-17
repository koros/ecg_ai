
from pathlib import Path
import sys

project_root = Path.cwd()

while not (project_root / "src").exists():
    project_root = project_root.parent

sys.path.insert(
    0,
    str(project_root)
)

print(f"project root found: {project_root}")


import time
from torch.utils.data import DataLoader
from src.dataset import ECGDataset





dataset = ECGDataset(
    "data/processed"
)

start = time.perf_counter()

for i in range(1000):
    dataset[i]

elapsed = time.perf_counter() - start

print(
    f"{1000/elapsed:.1f} samples/sec"
)



loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
)

start = time.time()

for batch_idx, batch in enumerate(loader):

    if batch_idx > 100:
        break

elapsed = time.time() - start

print(elapsed)

start = time.perf_counter()

n = 0

for x, y in loader:
    n += len(x)

elapsed = time.perf_counter() - start

print(
    f"{n/elapsed:.1f} samples/sec"
)


