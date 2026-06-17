import time

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

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import ECGDataset
from src.dataset import CachedECGDataset
from src.dataset import FastCachedECGDataset
from src.models import MLP
import argparse

def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    default="auto",
    choices=["auto", "cpu", "mps", "cuda"],
)
parser.add_argument(
    "--dataclass",
    default="fast",
    choices=["slow", "medium", "fast"],
)
parser.add_argument(
    "--profile",
    action="store_true",
)

args = parser.parse_args()

if args.device == "auto":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

else:
    device = torch.device(args.device)



if args.dataclass == 'fast':
    dataset = FastCachedECGDataset(
        "data/processed"
    )
elif args.dataclass == 'medium':
     dataset = CachedECGDataset(
        "data/processed"
    )
elif args.dataclass == 'slow':
    dataset = ECGDataset(
            "data/processed"
    )
n = len(dataset)

print(
    f"Training on {device} "
    f"with {len(dataset):,} samples "
    f"and dataclass: {args.dataclass}"
)

train_size = int(0.8 * n)
val_size = n - train_size

train_ds, val_ds = torch.utils.data.random_split(
    dataset,
    [train_size, val_size],
)


train_loader = DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=256,
)

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

epochs = 10
if args.profile:
    # detailed timing code
    start_training = time.perf_counter()

    load_time_total = 0.0
    transfer_time_total = 0.0
    forward_time_total = 0.0
    backward_time_total = 0.0
    optimizer_time_total = 0.0

    for epoch in range(epochs):

        model.train()
        
        running_loss = 0
        start = time.perf_counter()
        loader_iter = iter(train_loader)

        for batch_idx in range(len(train_loader)):

            loader_iter = iter(train_loader)


            #
            # 1. Batch loading
            #
            t0 = time.perf_counter()
            x, y = next(loader_iter)
            t1 = time.perf_counter()

            load_time_total += (t1 - t0)

            #
            # 2. CPU -> GPU transfer
            #
            sync_if_needed(device)

            t0 = time.perf_counter()
            x = x.to(device)
            y = y.to(device)
            sync_if_needed(device)

            t1 = time.perf_counter()

            transfer_time_total += (t1 - t0)

            optimizer.zero_grad()

            #
            # 3. Forward pass
            #
            sync_if_needed(device)
            t0 = time.perf_counter()
            logits = model(x)
            loss = criterion(
                logits,
                y,
                )

            sync_if_needed(device)

            t1 = time.perf_counter()

            forward_time_total += (t1 - t0)

            #
            # 4. Backward pass
            #
            sync_if_needed(device)
        
            t0 = time.perf_counter()

            loss.backward()
            sync_if_needed(device)
            t1 = time.perf_counter()
            backward_time_total += (t1 - t0)

            #
            # 5. Optimizer update
            #
            sync_if_needed(device)

            t0 = time.perf_counter()

            optimizer.step()

            sync_if_needed(device)

            t1 = time.perf_counter()

            optimizer_time_total += (t1 - t0)

            running_loss += loss.item()

            elapsed = (
                time.perf_counter() - start
                )

        print(
                f"Epoch {epoch+1} "
                f"loss={running_loss:.3f} "
                f"time={elapsed:.2f}s"
                )


        model.eval()




    elapsed_training = time.perf_counter() - start_training 
    print(
            f"\nEpoch {epoch+1}"
            )

    print(
            f"Loss                 : "
            f"{running_loss:.3f}"
            )

    print(
            f"Epoch time           : "
            f"{elapsed:.2f}s"
            )

    print(
            f"Batch loading        : "
            f"{load_time_total:.2f}s"
            )

    print(
            f"CPU->Device transfer : "
            f"{transfer_time_total:.2f}s"
            )

    print(
            f"Forward pass         : "
            f"{forward_time_total:.2f}s"
            )

    print(
            f"Backward pass        : "
            f"{backward_time_total:.2f}s"
            )

    print(
            f"Optimizer step       : "
            f"{optimizer_time_total:.2f}s"
            )


    total_profiled = (
            load_time_total
            + transfer_time_total
            + forward_time_total
            + backward_time_total
            + optimizer_time_total
            )
    print("\nBreakdown")

    print(
            f"Loading     : "
            f"{100*load_time_total/total_profiled:.1f}%"
            )

    print(
            f"Transfer    : "
            f"{100*transfer_time_total/total_profiled:.1f}%"
            )

    print(
            f"Forward     : "
            f"{100*forward_time_total/total_profiled:.1f}%"
            )

    print(
            f"Backward    : "
            f"{100*backward_time_total/total_profiled:.1f}%"
            )

    print(
            f"Optimizer   : "
            f"{100*optimizer_time_total/total_profiled:.1f}%"
            )

else:
    start_training = time.perf_counter()

    for epoch in range(epochs):

        model.train()
        
        running_loss = 0
        start = time.perf_counter()

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(
                logits,
                y,
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        elapsed = (
            time.perf_counter() - start
        )

        print(
            f"Epoch {epoch+1} "
            f"loss={running_loss:.3f} "
            f"time={elapsed:.2f}s"
        )


        model.eval()


    elapsed_training = time.perf_counter() - start_training 

correct = 0
total = 0

with torch.no_grad():

    for x, y in val_loader:

        x = x.to(device)
        y = y.to(device)

        pred = model(x).argmax(1)

        correct += (
                pred == y
                ).sum().item()

        total += len(y)

print(f"Accuracy: {correct/total:.3f}")
