from pathlib import Path
import sys
import time

project_root = Path.cwd()

while not (project_root / "src").exists():
    project_root = project_root.parent

sys.path.insert(
    0,
    str(project_root),
)

from torch.utils.data import DataLoader

from src.dataset import ECGDataset


dataset = ECGDataset(
    str(project_root)+"/data/processed"
)

print(
    f"Dataset size: {len(dataset):,}"
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

print("\n=== Slow Dataset ===")
print(f"Batches processed : {batch_idx+1}")
print(f"Samples processed : {samples_seen:,}")
print(f"Elapsed time      : {elapsed:.2f} s")
print(
    f"Throughput        : "
    f"{samples_seen/elapsed:.1f} samples/s"
)


