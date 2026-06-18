"""
plot_resources.py

Loads a CSV produced by resource_monitor.py and plots CPU, RAM, and
(if present) per-GPU utilization/memory/temperature/power over time.

Designed to be used in a notebook, either as:

    %run plot_resources.py path/to/run1.csv

or, more idiomatically inside a notebook cell:

    from plot_resources import plot_resources
    plot_resources("run1.csv")

Adapts automatically to however many GPUs are in the CSV (0, 1, or N) --
it just looks for gpu0_*, gpu1_*, etc. columns and plots whatever exists.
"""

import re
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load_resource_csv(path):
    """Load the CSV and parse the timestamp column into datetimes."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def detect_gpu_indices(df):
    """Find which GPU indices are present, e.g. [0, 1] for gpu0_*, gpu1_*."""
    indices = set()
    for col in df.columns:
        m = re.match(r"gpu(\d+)_", col)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def plot_resources(path, figsize_per_row=(10, 2.5)):
    """Load a resource_monitor.py CSV and plot everything in it.

    Produces one figure with stacked subplots:
      - CPU utilization (+ per-process CPU if present)
      - RAM usage
      - Thread count (if per-process tracking was enabled)
      - One panel per GPU metric (util, memory, temp, power), each
        panel overlaying all detected GPUs for easy comparison

    Returns the matplotlib Figure so it can be saved or tweaked further.
    """
    df = load_resource_csv(path)
    gpu_indices = detect_gpu_indices(df)
    has_proc = "proc_cpu_pct" in df.columns

    t = df["timestamp"]
    elapsed_min = (t - t.iloc[0]).dt.total_seconds() / 60.0

    panels = []  # list of (title, ylabel, plot_fn)

    def panel_cpu(ax):
        ax.plot(elapsed_min, df["cpu_pct_total"], label="node CPU %")
        if has_proc:
            ax.plot(elapsed_min, df["proc_cpu_pct"], label="process CPU %", linestyle="--")
        ax.set_ylabel("CPU %")
        ax.legend(loc="upper right", fontsize=8)
    panels.append(("CPU utilization", panel_cpu))

    def panel_ram(ax):
        ax.plot(elapsed_min, df["ram_used_gb"], label="RAM used (GB)")
        if has_proc and "proc_rss_gb" in df.columns:
            ax.plot(elapsed_min, df["proc_rss_gb"], label="process RSS (GB)", linestyle="--")
        ax.set_ylabel("GB")
        ax.legend(loc="upper right", fontsize=8)
    panels.append(("Memory usage", panel_ram))

    if has_proc and "proc_num_threads" in df.columns:
        def panel_threads(ax):
            ax.plot(elapsed_min, df["proc_num_threads"], color="tab:green")
            ax.set_ylabel("# threads")
        panels.append(("Process thread count", panel_threads))

    if gpu_indices:
        gpu_metrics = [
            ("util_pct", "GPU utilization", "%"),
            ("mem_used_mb", "GPU memory used", "MB"),
            ("temp_c", "GPU temperature", "°C"),
            ("power_w", "GPU power draw", "W"),
        ]
        for metric_key, title, ylabel in gpu_metrics:
            cols = [f"gpu{i}_{metric_key}" for i in gpu_indices if f"gpu{i}_{metric_key}" in df.columns]
            if not cols:
                continue

            def make_panel(cols=cols, ylabel=ylabel):
                def panel(ax):
                    for col in cols:
                        gpu_id = col.split("_")[0]
                        ax.plot(elapsed_min, df[col], label=gpu_id)
                    ax.set_ylabel(ylabel)
                    if len(cols) > 1:
                        ax.legend(loc="upper right", fontsize=8)
                return panel

            panels.append((title, make_panel()))

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(figsize_per_row[0], figsize_per_row[1] * n), sharex=True)
    if n == 1:
        axes = [axes]

    for (title, plot_fn), ax in zip(panels, axes):
        plot_fn(ax)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Elapsed time (minutes)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_resources.py path/to/resources.csv")
        sys.exit(1)

    fig = plot_resources(sys.argv[1])
    out_path = sys.argv[1].rsplit(".", 1)[0] + "_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()
