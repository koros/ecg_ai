import argparse
import csv
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
from src.models import build_model


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
    Note: each call costs a real torch.cuda.synchronize() round-trip
    (~microseconds) when profile=True; with many batches per epoch this
    adds up to a measurable (not dominant) fraction of total time -- see
    the --profile vs no --profile comparison if you want to quantify it
    on your own hardware.
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


def move_dataset_to_gpu(dataset, device):
    """Move an entire dataset's tensors to GPU memory once, up front.

    Only works for datasets that already expose .X and .y as full tensors
    in memory (FastCachedECGDataset / CachedECGDataset) -- this is exactly
    why --data_on_gpu is only meaningful with --dataclass fast or medium.
    ECGDataset (the --dataclass slow path) loads from disk per-sample and
    has no single tensor to move, so combining the two raises an error here.

    After this call, __getitem__ on the underlying dataset returns tensors
    that are already on the GPU. The existing .to(device, ...) calls in
    train_one_epoch / evaluate become no-ops (same tensor, zero copy) when
    --manual_batching is off; with --manual_batching on, the DataLoader is
    bypassed entirely (see train_one_epoch).

    Requires the entire dataset fit in GPU VRAM. For an ECG dataset on an
    H100 (80GB) this is almost always true; check resource_monitor.py's
    gpu0_mem_used_mb column if you suspect otherwise.
    """
    if not hasattr(dataset, "X") or not hasattr(dataset, "y"):
        raise ValueError(
            "--data_on_gpu requires a dataset exposing .X and .y tensors "
            "(FastCachedECGDataset or CachedECGDataset). "
            "--dataclass slow (ECGDataset) loads per-sample from disk and "
            "has no single tensor to move -- use --dataclass fast or medium instead."
        )
    dataset.X = dataset.X.to(device)
    dataset.y = dataset.y.to(device)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dataclass", default="fast", choices=["slow", "medium", "fast"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)

    # --- model selection ---
    parser.add_argument("--model", default="big", choices=["tiny", "big","huge"],
                        help="'tiny': original 2560->128->2 MLP (~328K params). "
                             "'big' (default): 2560->2048x4->2 MLP (~17.85M params), "
                             "'huge'  ~280M params), ")

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

    # --- DataLoader options ---
    parser.add_argument("--num_workers", type=int, default=0,
                    help="number of DataLoader worker processes for background batch loading "
                         "(default: 0 -- all loading on main process, blocking the GPU; "
                         "set to match your CPU allocation e.g. --num_workers 4). "
                         "Ignored (forced to 0) when --data_on_gpu is set.")
    parser.add_argument("--pin_memory", action="store_true",
                    help="allocate CPU batches in page-locked memory for faster CPU->GPU transfer "
                         "(use with --num_workers > 0 and non_blocking=True transfers; "
                         "no benefit without both). Ignored (forced off) when --data_on_gpu is set.")

    # --- GPU-resident dataset ---
    parser.add_argument("--data_on_gpu", action="store_true",
                    help="copy the entire dataset to GPU memory once at startup, instead of "
                         "transferring each batch from CPU every step. Only works with "
                         "--dataclass fast or medium (in-memory tensor datasets); raises "
                         "an error with --dataclass slow. Only useful if the full dataset "
                         "fits in GPU VRAM. Forces --num_workers to 0 and --pin_memory off "
                         "(both crash otherwise: DataLoader workers can't share GPU tensors, "
                         "and pin_memory only applies to CPU tensors).")
    parser.add_argument("--manual_batching", action="store_true",
                    help="requires --data_on_gpu. Bypasses the DataLoader entirely and "
                         "indexes batches directly out of dataset.X/.y using a GPU-resident "
                         "random permutation (torch.randperm(..., device=device)). Removes "
                         "per-sample Python __getitem__ and collate_fn overhead, which "
                         "remains even with --data_on_gpu alone (the DataLoader still "
                         "iterates sample-by-sample in Python before stacking into a "
                         "batch). This is the option to try if --data_on_gpu's 'transfer' "
                         "stage reads ~0%% but 'load' is still the dominant cost in "
                         "--profile output -- that combination means the bottleneck is "
                         "DataLoader/Python iteration overhead, not data movement, and "
                         "only --manual_batching addresses it.")

    # --- benchmarking / orchestration ---
    parser.add_argument("--result_csv", default=None,
                        help="if set, append a single row of run config + results to this "
                             "CSV (creating it with a header if it doesn't exist yet). "
                             "Intended for orchestration tools (Snakemake, benchmark "
                             "scripts) so they don't need to scrape stdout for timing/"
                             "accuracy -- the script writes its own structured result.")
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
                    profile, mixed_precision, amp_dtype, pin_memory,
                    manual_batching=False, train_dataset=None, batch_size=None):
    """Run one training epoch.

    When profile=True, accumulate time spent in each stage
    (loading / transfer / forward / backward / optimizer step).
    mixed_precision and amp_dtype control torch.autocast on the forward pass;
    when mixed_precision=False, autocast is a no-op (enabled=False).

    When manual_batching=True, the DataLoader (train_loader) is bypassed
    entirely: batches are built by indexing directly into train_dataset.X
    and train_dataset.y using a GPU-resident random permutation. This
    requires train_dataset and batch_size to be provided, and requires
    train_dataset.X / .y to already be tensors on `device` (i.e. produced
    by move_dataset_to_gpu). The "load" stage timing then measures the
    cost of the GPU-side gather (dataset.X[idx]) instead of DataLoader
    iteration -- compare against a non-manual-batching --data_on_gpu run
    to see the difference this makes.

    When manual_batching=False, the CPU->device transfer step is always
    called, even with --data_on_gpu -- but .to(device) on a tensor already
    on that device is a recognized no-op in PyTorch (same tensor returned,
    no copy), so the transfer stage's timing correctly drops to near zero
    without needing a separate code path for that case.

    Returns (running_loss, elapsed, timing).
    """
    if manual_batching and (train_dataset is None or batch_size is None):
        raise ValueError("--manual_batching requires train_dataset and batch_size "
                         "to be passed into train_one_epoch")

    model.train()
    running_loss = 0.0
    timing = {"load": 0.0, "transfer": 0.0, "forward": 0.0,
              "backward": 0.0, "optimizer": 0.0}

    start = time.perf_counter()

    if manual_batching:
        n_samples = train_dataset.X.shape[0]
        perm = torch.randperm(n_samples, device=device)  # shuffle, stays on GPU
        num_batches = (n_samples + batch_size - 1) // batch_size
    else:
        loader_iter = iter(train_loader)  # created once per epoch, not once per batch
        num_batches = len(train_loader)

    for batch_idx in range(num_batches):

        # 1. Batch loading
        t0 = maybe_sync_and_time(device, profile)
        if manual_batching:
            idx = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            x = train_dataset.X[idx]
            y = train_dataset.y[idx]
        else:
            x, y = next(loader_iter)
        t1 = maybe_sync_and_time(device, profile)
        if profile:
            timing["load"] += t1 - t0

        # 2. CPU -> device transfer (no-op if x, y are already on `device`,
        # which is always true when manual_batching=True)
        t0 = maybe_sync_and_time(device, profile)
        x = x.to(device, non_blocking=pin_memory)
        y = y.to(device, non_blocking=pin_memory)
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

    Note: always uses the DataLoader (train_loader), not manual_batching --
    if you want to profile the manual-batching path's kernels specifically,
    this would need a parallel code path; not added here since the
    --profile breakdown already isolates that comparison numerically.
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


def write_result_csv(path, args, final_epoch_time, accuracy):
    """Append one row of run config + results to a CSV for orchestration tools.

    Writes the header only if the file doesn't already exist yet, so this
    is safe to call from many independent job invocations writing to the
    same shared path (e.g. one rule per Snakemake job, each appending one
    row to a common results CSV). Concurrent writes from truly simultaneous
    jobs aren't locked here -- if you run many configs in parallel against
    the same --result_csv, prefer one CSV per job (as Snakemake's wildcard-
    based output naming does) and concatenate them in a separate step,
    rather than relying on this function's append to be safe under
    concurrent access.
    """
    fieldnames = [
        "model", "dataclass", "batch_size", "num_workers", "pin_memory",
        "data_on_gpu", "manual_batching", "tf32", "precision", "epochs",
        "epoch_time_s", "accuracy",
    ]
    precision = args.amp_dtype if args.mixed_precision else "fp32"
    row = {
        "model": args.model,
        "dataclass": args.dataclass,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "data_on_gpu": args.data_on_gpu,
        "manual_batching": args.manual_batching,
        "tf32": args.tf32,
        "precision": precision,
        "epochs": args.epochs,
        "epoch_time_s": f"{final_epoch_time:.4f}",
        "accuracy": f"{accuracy:.4f}",
    }

    path = Path(path)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Result row appended to {path}")


def print_run_config(args, device):
    """Print every CLI option actually in effect for this run, in one
    block, at the very start -- makes log/output files self-describing
    without needing to scroll back to find the original command line.

    Printed after device resolution (so --device auto shows the actual
    resolved device, e.g. "cuda" not "auto") but before any --data_on_gpu
    forcing of num_workers/pin_memory, so this reflects what was actually
    *requested* on the command line, not the post-correction values --
    if data_on_gpu later forces num_workers to 0, that's a separate
    printed line at the point it happens, not hidden here.
    """
    print("\n" + "=" * 64)
    print("Run configuration")
    print("=" * 64)
    print(f"  device              : {device}  (--device {args.device})")
    print(f"  model               : {args.model}")
    print(f"  dataclass           : {args.dataclass}")
    print(f"  epochs              : {args.epochs}")
    print(f"  batch_size          : {args.batch_size}")
    print(f"  num_workers         : {args.num_workers}")
    print(f"  pin_memory          : {args.pin_memory}")
    print(f"  data_on_gpu         : {args.data_on_gpu}")
    print(f"  manual_batching     : {args.manual_batching}")
    print(f"  tf32                : {args.tf32}")
    print(f"  mixed_precision     : {args.mixed_precision}")
    print(f"  amp_dtype           : {args.amp_dtype}"
          f"{' (inactive, mixed_precision is off)' if not args.mixed_precision else ''}")
    print(f"  profile             : {args.profile}")
    print(f"  torch_profile       : {args.torch_profile}")
    if args.torch_profile:
        print(f"  torch_profile_dir   : {args.torch_profile_dir}")
        print(f"  torch_profile_steps : {args.torch_profile_steps}")
    print(f"  result_csv          : {args.result_csv}")
    print("=" * 64 + "\n")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print_run_config(args, device)

    if args.manual_batching and not args.data_on_gpu:
        raise SystemExit("--manual_batching requires --data_on_gpu "
                         "(it indexes directly into GPU-resident dataset.X/.y)")

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

    # GPU-resident dataset: move the whole thing to GPU memory once, before
    # the train/val split and DataLoader construction, so both splits get
    # the benefit. num_workers/pin_memory are force-corrected here since
    # both crash (not just underperform) when combined with GPU-resident
    # tensors -- see move_dataset_to_gpu's docstring and parse_args' help.
    if args.data_on_gpu:
        dataset = move_dataset_to_gpu(dataset, device)
        print(f"Entire dataset moved to {device} "
              f"({dataset.X.element_size() * dataset.X.nelement() / 1e9:.2f} GB)")
        if args.num_workers > 0:
            print(f"--data_on_gpu set: forcing --num_workers 0 "
                  f"(was {args.num_workers}; DataLoader workers are separate "
                  f"processes and cannot safely access GPU tensors owned by "
                  f"the main process -- this combination crashes with a CUDA "
                  f"initialization error, not just inefficiency)")
            args.num_workers = 0
        if args.pin_memory:
            print(f"--data_on_gpu set: forcing --pin_memory off "
                  f"(pin_memory only applies to CPU tensors; calling it on "
                  f"GPU-resident tensors crashes with "
                  f"'cannot pin torch.cuda.FloatTensor')")
            args.pin_memory = False

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # train_loader is still constructed even when --manual_batching is set,
    # since run_torch_profiler (--torch_profile) always uses it regardless.
    # It's simply not iterated by train_one_epoch when manual_batching=True.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers,
                          pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=args.pin_memory)

    # manual_batching indexes train_ds (the Subset from random_split)
    # directly via train_ds.dataset.X/.y and train_ds.indices -- but Subset
    # doesn't expose .X/.y itself, so we build a small wrapper-free path:
    # gather the split's absolute indices once as a GPU tensor, then index
    # the underlying dataset's full X/y with a permutation over *those*
    # positions (not the full dataset) inside train_one_epoch. To keep
    # train_one_epoch simple, we pass it an object exposing .X/.y that are
    # already restricted to the training split.
    train_dataset_for_manual_batching = None
    if args.manual_batching:
        train_indices = torch.as_tensor(train_ds.indices, device=device)
        class _SplitView:
            pass
        train_dataset_for_manual_batching = _SplitView()
        train_dataset_for_manual_batching.X = dataset.X[train_indices]
        train_dataset_for_manual_batching.y = dataset.y[train_indices]

    model = build_model(args.model).to(device)
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
            args.profile, args.mixed_precision, amp_dtype, args.pin_memory,
            manual_batching=args.manual_batching,
            train_dataset=train_dataset_for_manual_batching,
            batch_size=args.batch_size,
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

    if args.result_csv:
        write_result_csv(args.result_csv, args, elapsed, accuracy)


if __name__ == "__main__":
    main()
