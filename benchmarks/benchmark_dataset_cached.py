from pathlib import Path
import sys
import time

from torch.utils.data import DataLoader

project_root = Path.cwd()

while not (project_root / "src").exists():
    project_root = project_root.parent

sys.path.insert(
    0,
    str(project_root),
)

from src.dataset import CachedECGDataset
def benchmark(loader, name):

    start = time.time()

    samples_seen = 0

    for batch_idx, (x, y) in enumerate(loader):

        samples_seen += len(x)

        if batch_idx >= 100:
            break

    elapsed = time.time() - start

    print(f"\n=== {name} ===")
    print(f"Samples processed : {samples_seen:,}")
    print(f"Elapsed time      : {elapsed:.2f} s")
    print(
        f"Throughput        : "
        f"{samples_seen/elapsed:.1f} samples/s"
    )


def main():
    dataset = CachedECGDataset(
            "data/processed"
            )

    benchmark(
            DataLoader(
                dataset,
                batch_size=128,
                num_workers=0,
                ),
            "Cached Dataset (0 workers)"
            )

    benchmark(
            DataLoader(
                dataset,
                batch_size=128,
                num_workers=2,
                ),
            "Cached Dataset (2 workers)"
            )

    benchmark(
            DataLoader(
                dataset,
                batch_size=128,
                num_workers=4,
                ),
            "Cached Dataset (4 workers)"
            )


if __name__ == "__main__":
    main()
