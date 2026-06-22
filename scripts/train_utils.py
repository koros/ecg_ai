"""
train_utils.py

Shared infrastructure used by both train_resnet1d.py and train_dual.py:
DDP setup, GPU-resident dataset loading, manual rank-aware batching,
result-CSV writer, run-config printing, and the actual training loop
(which is architecture-agnostic -- the two scripts only differ in which
model class they instantiate, and how it's parameterized via CLI flags).

Distributed semantics:
    - torchrun-launched: each rank is its own process with LOCAL_RANK env
    - Each rank pins to its own GPU (cuda:LOCAL_RANK)
    - Dataset is loaded once per rank but copied to each rank's GPU
      (NVLink-equipped nodes do this fast; on PCIe nodes this is a tax
      paid once at startup, not per-step, so it's fine)
    - Strong scaling: --batch_size is GLOBAL; each rank processes
      batch_size/world_size samples per step
    - Manual batching: each rank generates a GPU-resident permutation
      over its own non-overlapping slice of the training indices
    - Only rank 0 prints / writes the result CSV
"""

import csv
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn


# ---------------------------------------------------------------------
# DDP setup / teardown
# ---------------------------------------------------------------------

def setup_distributed():
    """Initialize torch.distributed if launched under torchrun, else
    run in single-process mode. Returns (rank, world_size, local_rank,
    device). On a single process, world_size=1 and dist is not initialized
    -- the rest of the code in this module guards on `world_size > 1` for
    any collective calls (all_reduce, barrier), so single-GPU runs work
    without torchrun.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def is_main_rank(rank):
    return rank == 0


def print_main(rank, *args, **kwargs):
    """print() that only fires on rank 0 -- avoids 4x-duplicated output
    on a 4-GPU run."""
    if is_main_rank(rank):
        print(*args, **kwargs)


# ---------------------------------------------------------------------
# Dataset loading + GPU residency
# ---------------------------------------------------------------------

def find_project_root():
    """Walk up from cwd until we find a 'src' subdir -- matches the
    existing convention used by all the training scripts in this repo."""
    project_root = Path.cwd()
    while not (project_root / "src").exists():
        if project_root == project_root.parent:
            raise RuntimeError("Could not find project root (no 'src' dir in any ancestor)")
        project_root = project_root.parent
    return project_root


def build_dataset(name, project_root):
    """Construct dataset by name. Only 'fast' and 'medium' are valid
    here -- 'slow' (per-sample disk loading) doesn't make sense once
    we're going GPU-resident."""
    # Import inside the function so this module doesn't fail to import
    # on systems without the project's src/ in sys.path yet.
    from src.dataset import CachedECGDataset, FastCachedECGDataset

    data_dir = project_root / "data" / "processed"
    if name == "fast":
        return FastCachedECGDataset(data_dir)
    elif name == "medium":
        return CachedECGDataset(data_dir)
    raise ValueError(f"unknown dataclass: {name!r} (use 'fast' or 'medium')")


def move_dataset_to_gpu(dataset, device):
    """Move .X and .y to the given GPU. Required for manual batching."""
    if not hasattr(dataset, "X") or not hasattr(dataset, "y"):
        raise ValueError(
            "Dataset must expose .X and .y tensors (FastCachedECGDataset "
            "or CachedECGDataset). 'slow' dataclass is not supported here."
        )
    dataset.X = dataset.X.to(device)
    dataset.y = dataset.y.to(device)
    return dataset


def make_split_view(dataset, indices, device):
    """Build a lightweight object exposing .X / .y restricted to the
    given indices (a CPU list of ints from random_split). Returns an
    object with .X and .y already on `device`. Same _SplitView pattern
    used in train_MLP_final3.py, factored out here so both training
    scripts can share it.
    """
    indices_t = torch.as_tensor(indices, device=device)

    class _SplitView:
        pass

    view = _SplitView()
    view.X = dataset.X[indices_t]
    view.y = dataset.y[indices_t]
    return view


def partition_for_rank(split_view, rank, world_size, device):
    """Slice the training split into one non-overlapping chunk per DDP
    rank. Returns a _SplitView restricted to that rank's chunk.

    Strong-scaling semantics: each rank processes 1/world_size of the
    training data per epoch, so global per-epoch work is constant
    regardless of GPU count.

    Note: this drops the trailing remainder if n_samples is not evenly
    divisible by world_size, to keep all ranks on identical-sized work
    (avoids one rank lagging at the end of each epoch and triggering
    all-reduce timeouts). For 705,732 training samples / 4 ranks =
    176,433 per rank, dropping at most 3 samples -- negligible for
    scaling measurements.
    """
    n = split_view.X.shape[0]
    per_rank = n // world_size
    start = rank * per_rank
    end = start + per_rank

    class _RankShard:
        pass

    shard = _RankShard()
    shard.X = split_view.X[start:end]
    shard.y = split_view.y[start:end]
    return shard


# ---------------------------------------------------------------------
# Training loop (manual batching, DDP-aware)
# ---------------------------------------------------------------------

def train_one_epoch(model, optimizer, criterion, device, rank_shard,
                    per_rank_batch_size, mixed_precision, amp_dtype,
                    world_size):
    """Run one epoch of manual-batched training on a single rank's data.

    Each rank shuffles its own shard (independent perm per rank, since
    DDP's gradient all_reduce handles the cross-rank averaging anyway --
    we just need each rank to see its own samples in some order).
    """
    model.train()
    running_loss = 0.0
    start = time.perf_counter()

    n = rank_shard.X.shape[0]
    perm = torch.randperm(n, device=device)
    num_batches = n // per_rank_batch_size  # drop incomplete final batch

    for batch_idx in range(num_batches):
        idx = perm[batch_idx * per_rank_batch_size:
                   (batch_idx + 1) * per_rank_batch_size]
        x = rank_shard.X[idx]
        y = rank_shard.y[idx]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                           enabled=mixed_precision):
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Force any queued GPU work to finish before timing, so per-epoch
    # timing reflects actual GPU work, not just CPU dispatch time.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_local = time.perf_counter() - start

    # Across-rank timing: report the slowest rank's epoch time
    # (true wall-clock since all ranks must finish before next epoch).
    if world_size > 1:
        t = torch.tensor([elapsed_local], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        elapsed = t.item()
    else:
        elapsed = elapsed_local

    return running_loss, elapsed


def evaluate(model, val_dataset, device, batch_size):
    """Validation accuracy on rank 0 only. Other ranks skip this.
    Uses manual batching against the GPU-resident val_dataset.X/.y.
    """
    model.eval()
    correct = 0
    total = 0
    n = val_dataset.X.shape[0]
    num_batches = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            x = val_dataset.X[batch_idx * batch_size:
                              (batch_idx + 1) * batch_size]
            y = val_dataset.y[batch_idx * batch_size:
                              (batch_idx + 1) * batch_size]
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


# ---------------------------------------------------------------------
# Result CSV writer (rank 0 only)
# ---------------------------------------------------------------------

def write_result_csv(path, fieldnames, row):
    """Append one structured row to a CSV. Writes the header if file
    doesn't exist yet. Caller is responsible for ensuring this is only
    called from rank 0 (we do NOT check that here -- check at call site).
    """
    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------
# Run-config printer
# ---------------------------------------------------------------------

def print_run_config(args, world_size, device, extra_lines=None):
    """Print every CLI option in effect, plus DDP topology, in one block
    at the start of the run. `extra_lines` is a list of (label, value)
    tuples for architecture-specific fields that aren't part of the
    shared CLI (e.g. resnet width/depth vs dual-branch hyperparameters).
    """
    print("=" * 64)
    print("Run configuration")
    print("=" * 64)
    print(f"  world_size          : {world_size}")
    print(f"  device              : {device}")
    print(f"  dataclass           : {args.dataclass}")
    print(f"  epochs              : {args.epochs}")
    print(f"  batch_size (global) : {args.batch_size}")
    print(f"  per-rank batch      : {args.batch_size // world_size}")
    print(f"  tf32                : {args.tf32}")
    print(f"  mixed_precision     : {args.mixed_precision}")
    print(f"  amp_dtype           : {args.amp_dtype}"
          f"{' (inactive)' if not args.mixed_precision else ''}")
    if extra_lines:
        for label, value in extra_lines:
            print(f"  {label:20s}: {value}")
    print(f"  result_csv          : {args.result_csv}")
    print("=" * 64)


# ---------------------------------------------------------------------
# Common CLI args added to both training scripts
# ---------------------------------------------------------------------

def add_common_args(parser):
    """Add the CLI flags that both training scripts share. Architecture-
    specific flags (--width, --depth, etc.) are added separately by each
    script.
    """
    parser.add_argument("--dataclass", default="fast", choices=["fast", "medium"],
                        help="dataset variant -- 'fast' (in-memory) recommended for scaling work")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8192,
                        help="GLOBAL batch size across all DDP ranks. Each rank processes "
                             "batch_size/world_size samples per step (strong scaling).")
    parser.add_argument("--tf32", action="store_true",
                        help="enable TF32 for fp32 matmul/cuDNN ops on Ampere+ GPUs")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="wrap forward pass in torch.autocast (bf16 by default)")
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"],
                        help="dtype for autocast when --mixed_precision is on")
    parser.add_argument("--result_csv", default=None,
                        help="if set, append a structured row of config + results to this CSV "
                             "(rank 0 only)")
