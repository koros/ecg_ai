"""
plot_benchmark.py

Notebook-friendly plotting tools for the benchmark CSV produced by the
Snakemake workflow + train_MLP_final3.py's --result_csv option.

Usage in a notebook:

    from plot_benchmark import plot_all
    fig = plot_all("benchmark_combined.csv")

or individual panels:

    from plot_benchmark import (
        load_benchmark, plot_dataloader_tuning, plot_data_on_gpu_progression,
        plot_precision_vs_batch_size, plot_amdahl_comparison,
    )
    df = load_benchmark("benchmark_combined.csv")
    plot_dataloader_tuning(df)

The four panels each tell one self-contained part of the story:
  1. DataLoader tuning -- num_workers + pin_memory effect
  2. data_on_gpu -> manual_batching progression at bs=256
  3. Precision flags (tf32, bf16) at bs=256 vs bs=8192 for the big model
  4. Amdahl comparison -- big vs huge at small vs large batch

Each function returns its matplotlib Figure for further tweaking/saving.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_benchmark(path):
    """Load the combined benchmark CSV with sensible dtypes."""
    df = pd.read_csv(path)
    # Booleans-as-strings can happen depending on how pandas reads --
    # normalize them here so downstream filtering is consistent.
    for col in ["pin_memory", "data_on_gpu", "manual_batching", "tf32"]:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False}).astype(bool)
    return df


def plot_dataloader_tuning(df, ax=None):
    """Panel 1: DataLoader tuning at bs=256, no data_on_gpu.

    Shows the effect of num_workers (0/2/4) and pin_memory (yes/no) on
    epoch time. dataclass=fast only (medium follows the same pattern).
    Demonstrates the classic 'CPU-bound DataLoader' lesson: workers and
    pin_memory together give ~30%% speedup, but neither alone does much.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    sub = df[(df["dataclass"] == "fast") & (df["data_on_gpu"] == False)
            & (df["epochs"] == 3) & (df["manual_batching"] == False)].copy()

    workers = sorted(sub["num_workers"].unique())
    width = 0.35
    x = np.arange(len(workers))

    times_nopin = [sub[(sub["num_workers"] == w) & (sub["pin_memory"] == False)]["epoch_time_s"].mean()
                   for w in workers]
    times_pin = [sub[(sub["num_workers"] == w) & (sub["pin_memory"] == True)]["epoch_time_s"].mean()
                 for w in workers]

    ax.bar(x - width/2, times_nopin, width, label="pin_memory=False", color="#888")
    ax.bar(x + width/2, times_pin, width, label="pin_memory=True", color="#3a7")

    ax.set_xticks(x)
    ax.set_xticklabels([f"num_workers={w}" for w in workers])
    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Panel 1: DataLoader tuning (model=big, bs=256, dataclass=fast, no data_on_gpu)",
                fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_data_on_gpu_progression(df, ax=None):
    """Panel 2: progression from tuned DataLoader -> data_on_gpu -> manual_batching.

    At bs=256, model=big, fp32 -- shows that each architectural change
    removes another layer of overhead. Tuned DataLoader -> data_on_gpu
    eliminates CPU transfer; data_on_gpu -> manual_batching eliminates
    Python-level DataLoader iteration. Together, ~40%% speedup.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    # All at bs=256, model=big, fp32, no tf32, epochs=10
    base = df[(df["batch_size"] == 256) & (df["model"] == "big")
              & (df["epochs"] == 10) & (df["precision"] == "fp32")
              & (df["tf32"] == False)]

    configs = [
        ("baseline\n(nw=0, no pin)",
         base[(base["num_workers"] == 0) & (base["pin_memory"] == False)
              & (base["data_on_gpu"] == False) & (base["manual_batching"] == False)]),
        ("tuned DataLoader\n(nw=4, pin)",
         base[(base["num_workers"] == 4) & (base["pin_memory"] == True)
              & (base["data_on_gpu"] == False) & (base["manual_batching"] == False)]),
        ("data_on_gpu\n(no manual)",
         base[(base["data_on_gpu"] == True) & (base["manual_batching"] == False)]),
        ("data_on_gpu\n+ manual_batching",
         base[(base["data_on_gpu"] == True) & (base["manual_batching"] == True)]),
    ]

    labels = [c[0] for c in configs]
    times = [c[1]["epoch_time_s"].mean() if len(c[1]) else np.nan for c in configs]

    bars = ax.bar(labels, times, color=["#888", "#5a8", "#3a7", "#1a5"])
    for bar, t in zip(bars, times):
        if not np.isnan(t):
            ax.text(bar.get_x() + bar.get_width()/2, t + 0.1, f"{t:.2f}s",
                   ha="center", fontsize=9)

    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Panel 2: removing overhead layers (model=big, bs=256, fp32)",
                fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_precision_vs_batch_size(df, ax=None):
    """Panel 3: precision flags on `big` model at bs=256 vs bs=8192.

    All with data_on_gpu + manual_batching (so the loop is as lean as
    possible). Shows the punchline: TF32/bf16 are noise at bs=256 but
    give 3-5x speedups at bs=8192 -- because the GPU was idle most of
    the time at bs=256, so optimizing compute couldn't help.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    sub = df[(df["model"] == "big") & (df["data_on_gpu"] == True)
             & (df["manual_batching"] == True) & (df["epochs"] == 10)].copy()

    configs = [
        ("fp32",          {"tf32": False, "precision": "fp32"}),
        ("tf32",          {"tf32": True,  "precision": "fp32"}),
        ("bf16",          {"tf32": False, "precision": "bf16"}),
        ("tf32 + bf16",   {"tf32": True,  "precision": "bf16"}),
    ]

    width = 0.35
    x = np.arange(len(configs))

    times_256 = []
    times_8192 = []
    for _, cfg in configs:
        match256 = sub[(sub["batch_size"] == 256) & (sub["tf32"] == cfg["tf32"])
                       & (sub["precision"] == cfg["precision"])]
        match8192 = sub[(sub["batch_size"] == 8192) & (sub["tf32"] == cfg["tf32"])
                        & (sub["precision"] == cfg["precision"])]
        times_256.append(match256["epoch_time_s"].mean() if len(match256) else np.nan)
        times_8192.append(match8192["epoch_time_s"].mean() if len(match8192) else np.nan)

    ax.bar(x - width/2, times_256, width, label="bs=256", color="#c66")
    ax.bar(x + width/2, times_8192, width, label="bs=8192", color="#37a")

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in configs])
    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Panel 3: precision flags help only when GPU is busy (model=big, data_on_gpu, manual_batching)",
                fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_amdahl_comparison(df, ax=None):
    """Panel 4: the Amdahl punchline -- model size + batch size are
    interchangeable levers on the same underlying variable
    (work per kernel launch).

    Compares big (17.85M params) vs huge (222M params) at bs=256 vs
    bs=8192, fp32 baseline vs best precision flags. The big model at
    bs=256 isn't compute-bound -> precision flags don't help.
    The huge model at bs=256 IS compute-bound -> precision flags
    finally give a real speedup at small batch size too.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    sub = df[(df["data_on_gpu"] == True) & (df["manual_batching"] == True)
             & (df["epochs"] == 10)].copy()

    # Four x-axis groups: (model, batch_size); two bars each (fp32 baseline, best=bf16+tf32)
    groups = [("big", 256), ("big", 8192), ("huge", 256), ("huge", 8192)]
    labels = [f"{m}\nbs={bs}" for (m, bs) in groups]
    width = 0.35
    x = np.arange(len(groups))

    fp32_times = []
    best_times = []
    for model, bs in groups:
        fp32_row = sub[(sub["model"] == model) & (sub["batch_size"] == bs)
                       & (sub["tf32"] == False) & (sub["precision"] == "fp32")]
        best_row = sub[(sub["model"] == model) & (sub["batch_size"] == bs)
                       & (sub["tf32"] == True) & (sub["precision"] == "bf16")]
        fp32_times.append(fp32_row["epoch_time_s"].mean() if len(fp32_row) else np.nan)
        best_times.append(best_row["epoch_time_s"].mean() if len(best_row) else np.nan)

    bars_fp32 = ax.bar(x - width/2, fp32_times, width, label="fp32 baseline", color="#888")
    bars_best = ax.bar(x + width/2, best_times, width, label="tf32 + bf16", color="#1a5")

    # Annotate speedup ratio on top of each pair
    for i, (fp32, best) in enumerate(zip(fp32_times, best_times)):
        if not (np.isnan(fp32) or np.isnan(best)) and best > 0:
            speedup = fp32 / best
            ax.text(i, max(fp32, best) + 1.0, f"{speedup:.1f}x",
                   ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Panel 4: precision flags help when work-per-launch is large (big->huge OR small bs->large bs)",
                fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_all(path, figsize=(12, 14)):
    """Render all four panels in one stacked figure.

    Returns the Figure so it can be saved or further tweaked in a notebook.
    """
    df = load_benchmark(path)
    fig, axes = plt.subplots(4, 1, figsize=figsize)

    plot_dataloader_tuning(df, ax=axes[0])
    plot_data_on_gpu_progression(df, ax=axes[1])
    plot_precision_vs_batch_size(df, ax=axes[2])
    plot_amdahl_comparison(df, ax=axes[3])

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_combined.csv"
    fig = plot_all(path)
    out = path.rsplit(".", 1)[0] + "_plots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
