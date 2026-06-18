import argparse
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = Path.cwd()
while not (project_root / "src").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
print(f"project root found: {project_root}")

from src.dataset import ECGDataset, CachedECGDataset, FastCachedECGDataset
from src.models import MLP


def sync_if_needed(device):
    """Block until pending GPU work finishes. No-op on CPU.

    Required before/after timing any GPU-bound section, since CUDA/MPS
    calls are launched asynchronously and return before the work is done.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def maybe_sync_and_time(device, profile):
    """Sync (if profiling) and return a timestamp; else return None.

    Called on both sides of each timed section. The branch and function
    call cost ~100ns, dwarfed by the synchronize() and the ms-scale GPU
    work being timed -- not a performance concern even per-batch.
    """
    if profile:
        sync_if_needed(device)
        return time.perf_counter()
    return None


def build_dataset(name):
    """Build dataset from project_root/data/processed using an absolute path.

    Uses project_root (resolved at module load time by walking up from cwd)
    so the script works regardless of which directory SLURM launches it from.
    Path joining uses / operator (not string +) since project_root is a Path.
    """
    data_dir = project_root / "data" / "processed"
    if name == "fast":
        return FastCachedECGDataset(data_dir)
    elif name == "medium":
        return CachedECGDataset(data_dir)
    elif name == "slow":
        return ECGDataset(data_dir)
    raise ValueError(f"unknown dataclass: {name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dataclass", default="fast", choices=["slow", "medium", "fast"])
    parser.add_argument("--epochs", type=int, default=10)

    # --- manual stage timing ---
    parser.add_argument("--profile", action="store_true",
                        help="manual stage-by-stage timing (load/transfer/forward/backward/opt)")

    # --- torch.profiler (per-op kernel traces for TensorBoard) ---
    parser.add_argument("--torch_profile", action="store_true",
                        help="run torch.profiler for a short pass before main training; "
                             "writes a trace directory readable by TensorBoard / Perfetto")
    parser.add_argument("--torch_profile_dir", default="torch_profiler_logs",
                        help="output directory for torch.profiler traces (default: torch_profiler_logs)")
    parser.add_argument("--torch_profile_steps", type=int, default=10,
                        help="number of active steps to profile (default: 10); "
                             "keep small -- traces grow fast and profiling adds overhead")

    # --- precision options ---
    parser.add_argument("--tf32", action="store_true",
                        help="enable TF32 for fp32 matmul and cuDNN ops on Ampere+ GPUs "
                             "(Tensor Core acceleration with minor precision tradeoff, no dtype change)")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="wrap forward pass in torch.autocast for Tensor Core acceleration "
                             "via reduced precision (bf16 by default, see --amp_dtype)")
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"],
                        help="dtype used by autocast when --mixed_precision is on "
                             "(default: bf16 -- same exponent range as fp32, safe without GradScaler; "
                             "fp16 has a smaller exponent range and needs GradScaler for real training)")
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    profile, mixed_precision, amp_dtype):
    """Run one training epoch.

    When profile=True, accumulate time spent in each stage
    (loading / transfer / forward / backward / optimizer step).
    mixed_precision and amp_dtype control torch.autocast on the forward pass;
    when mixed_precision=False, autocast is a no-op (enabled=False).
    Returns (running_loss, elapsed, timing).
    """
    model.train()
    running_loss = 0.0
    timing = {"load": 0.0, "transfer": 0.0, "forward": 0.0,
              "backward": 0.0, "optimizer": 0.0}

    start = time.perf_counter()
    loader_iter = iter(train_loader)  # created once per epoch, not once per batch

    for _ in range(len(train_loader)):

        # 1. Batch loading
        t0 = maybe_sync_and_time(device, profile)
        x, y = next(loader_iter)
        t1 = maybe_sync_and_time(device, profile)
        if profile:
            timing["load"] += t1 - t0

        # 2. CPU -> device transfer
        t0 = maybe_sync_and_time(device, profile)
        x = x.to(device)
        y = y.to(device)
        t1 = maybe_sync_and_time(device, profile)
        if profile:
            timing["transfer"] += t1 - t0

        optimizer.zero_grad()

        # 3. Forward pass
        # autocast is a no-op when enabled=False, so this is safe to leave
        # in permanently regardless of whether --mixed_precision is passed.
        t0 = maybe_sync_and_time(device, profile)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
            logits = model(x)
            loss = criterion(logits, y)
        t1 = maybe_sync_and_time(device, profile)
        if profile:
            timing["forward"] += t1 - t0

        # 4. Backward pass
        t0 = maybe_sync_and_time(device, profile)
        loss.backward()
        t1 = maybe_sync_and_time(device, profile)
        if profile:
            timing["backward"] += t1 - t0

        # 5. Optimizer step
        t0 = maybe_sync_and_time(device, profile)
        optimizer.step()
        t1 = maybe_sync_and_time(device, profile)
        if profile:
            timing["optimizer"] += t1 - t0

        running_loss += loss.item()

    elapsed = time.perf_counter() - start
    model.eval()
    return running_loss, elapsed, timing


def print_profile_breakdown(timing):
    total = sum(timing.values())
    print(f"Batch loading        : {timing['load']:.2f}s")
    print(f"CPU->Device transfer : {timing['transfer']:.2f}s")
    print(f"Forward pass         : {timing['forward']:.2f}s")
    print(f"Backward pass        : {timing['backward']:.2f}s")
    print(f"Optimizer step       : {timing['optimizer']:.2f}s")
    print("\nBreakdown")
    for label, key in [("Loading",   "load"),
                        ("Transfer",  "transfer"),
                        ("Forward",   "forward"),
                        ("Backward",  "backward"),
                        ("Optimizer", "optimizer")]:
        print(f"{label:12s}: {100 * timing[key] / total:.1f}%")


def run_torch_profiler(model, train_loader, optimizer, criterion, device,
                       out_dir, num_steps, mixed_precision, amp_dtype):
    """Profile a handful of training steps with torch.profiler.

    Uses the built-in schedule: 1 wait step, 1 warmup step, then num_steps
    active steps. Traces are written via tensorboard_trace_handler and can
    be opened with:  tensorboard --logdir <out_dir>
    or dragged into https://ui.perfetto.dev (Chrome only for TensorBoard).

    Runs as a short separate pass before the main training loop -- does not
    affect epoch counts or loss curves. Respects --mixed_precision/--amp_dtype
    so the profiled kernels match what the real training loop will run.
    """
    from torch.profiler import (profile, schedule,
                                tensorboard_trace_handler, ProfilerActivity)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    prof_schedule = schedule(wait=1, warmup=1, active=num_steps, repeat=1)
    model.train()
    loader_iter = iter(train_loader)

    print(f"\nRunning torch.profiler for {num_steps} active steps "
          f"(+1 wait, +1 warmup) -> writing trace to {out_dir}")

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(out_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(num_steps + 2):  # +1 wait, +1 warmup
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                x, y = next(loader_iter)

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                 enabled=mixed_precision):
                logits = model(x)
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            prof.step()

    print(f"Trace written to: {out_dir}")
    print(f"View with:  tensorboard --logdir {out_dir}")
    print(f"Or open in: https://ui.perfetto.dev  (drag-and-drop the .json file)")


def evaluate(model, val_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # TF32: global backend switch -- must be set before any matmul/conv op,
    # so it goes here, before the model and data loaders are created.
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for matmul and cuDNN ops")

    # Resolve amp dtype once; passed through to train_one_epoch and
    # run_torch_profiler so the profiled kernels match real training.
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    if args.mixed_precision:
        print(f"Mixed precision enabled (autocast dtype: {args.amp_dtype})")

    dataset = build_dataset(args.dataclass)
    print(f"Training on {device} with {len(dataset):,} samples "
          f"and dataclass: {args.dataclass}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.torch_profile:
        run_torch_profiler(
            model, train_loader, optimizer, criterion, device,
            args.torch_profile_dir, args.torch_profile_steps,
            args.mixed_precision, amp_dtype,
        )

    start_training = time.perf_counter()
    for epoch in range(args.epochs):
        running_loss, elapsed, timing = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            args.profile, args.mixed_precision, amp_dtype,
        )
        print(f"Epoch {epoch + 1} loss={running_loss:.3f} time={elapsed:.2f}s")

    elapsed_training = time.perf_counter() - start_training

    if args.profile:
        print(f"\nEpoch {args.epochs}")
        print(f"Loss                 : {running_loss:.3f}")
        print(f"Epoch time           : {elapsed:.2f}s")
        print_profile_breakdown(timing)

    accuracy = evaluate(model, val_loader, device)
    print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
