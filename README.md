# ecg_ai

PyTorch workshop project for ECG classification and GPU/HPC performance experiments.

The code uses the PhysioNet Long-Term AF Database (LTAFDB) to build fixed-length
ECG windows, label them for atrial fibrillation, and train several PyTorch
classifiers. The repository is set up to demonstrate the full path from data
preprocessing and baseline training through DataLoader tuning, GPU-resident
datasets, mixed precision, and single-node distributed data parallel scaling.

Slides for the workshop are in `Pytorch.pdf`.

## What is included

- `scripts/preprocess.py` and `scripts/preprocess_parallel.py`: convert raw
  LTAFDB WFDB records into compressed NumPy window files.
- `src/dataset.py`: three dataset implementations used to compare per-sample
  disk loading, cached NumPy loading, and prebuilt tensor loading.
- `src/models.py`: MLP baselines (`tiny`, `big`, `huge`).
- `src/models_resnet1d.py`: parameterized 1D ResNet for ECG signals.
- `src/models_dual.py`: dual-branch CNN + Transformer ECG classifier.
- `scripts/train_MLP.py`: simple MLP training script.
- `scripts/train_MLP_exp.py`: extended MLP benchmarking script with profiling,
  TensorBoard traces, TF32, AMP, DataLoader options, GPU-resident data, manual
  batching, and CSV result output.
- `scripts/train_resnet1d.py` and `scripts/train_dual.py`: DDP-aware training
  scripts for single-node multi-GPU scaling with `torchrun`.
- `slurm/`: SLURM job scripts and Snakemake workflows for benchmark sweeps.
- `src/plot_benchmark.py`, `src/plot_scaling.py`, and `src/plot_resources.py`:
  plotting helpers for benchmark and resource-monitor CSVs.
- `notebooks/visualise_data.ipynb`: exploratory notebook for inspecting data.

## Data

This project expects the PhysioNet LTAFDB records:

- https://www.physionet.org/content/ltafdb/1.0.0/
- https://archive.physionet.org/physiobank/database/ltafdb/

The expected raw-data layout is:

```text
data/raw/ltafdb/
  00.dat
  00.hea
  00.atr
  ...
```

On the workshop HPC system, the Makefile includes a convenience target that
copies prepared data from scratch storage:

```bash
make copy-data
```

For a local installation, use `wfdb` or another PhysioNet download method to
place the LTAFDB records under `data/raw/ltafdb`.

## Environment

The main Python dependencies are:

- Python 3.11
- PyTorch
- NumPy
- SciPy
- pandas
- matplotlib
- tqdm
- wfdb
- fsspec
- nvidia-ml-py
- tensorboard
- torch-tb-profiler
- snakemake, for the workflow files

On the target HPC environment, load the modules in `modules_for_ecg_ai`:

```bash
source modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate
```

The same environment is also described in `HPC_pytorch_AI.toml`.

For a local virtual environment, install the equivalent packages, for example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy pandas matplotlib tqdm wfdb fsspec nvidia-ml-py tensorboard torch-tb-profiler
```

Install the PyTorch build appropriate for your platform and CUDA version if the
default `pip install torch` is not suitable.

## Preprocessing

Preprocessing reads WFDB records and annotations, builds a per-sample rhythm
label array, extracts fixed-size ECG windows, and writes one compressed `.npz`
file per record.

Default settings:

- sampling rate: `128 Hz`
- window length: `10 s`
- stride: `10 s`
- input: `data/raw/ltafdb`
- output: `data/processed`
- label: `1` if a majority of the window is marked `(AFIB`, otherwise `0`

Run the serial preprocessor:

```bash
python scripts/preprocess.py \
  --data-dir data/raw/ltafdb \
  --output-dir data/processed \
  --window-seconds 10 \
  --stride-seconds 10
```

Run the parallel preprocessor:

```bash
python scripts/preprocess_parallel.py \
  --data-dir data/raw/ltafdb \
  --output-dir data/processed \
  --window-seconds 10 \
  --stride-seconds 10 \
  --workers 16
```

Each output file contains:

```text
X: (n_windows, 1280, 2) float32
y: (n_windows,) uint8
```

## Dataset Variants

Training scripts use `--dataclass` to select how windows are loaded:

- `slow`: `ECGDataset`, keeps `(file, index)` pairs and opens `.npz` files in
  `__getitem__`. Useful as a deliberately slow baseline.
- `medium`: `CachedECGDataset`, concatenates all NumPy arrays in memory and
  converts samples to tensors in `__getitem__`.
- `fast`: `FastCachedECGDataset`, concatenates all arrays and converts the full
  dataset to tensors at construction time. This is the recommended default for
  most GPU experiments.

The training scripts expect processed files in `data/processed`.

## Training

### Simple MLP baseline

```bash
python scripts/train_MLP.py --device auto --dataclass fast --epochs 10
```

Useful options:

```bash
python scripts/train_MLP.py --device cuda --dataclass fast --epochs 10 --profile
python scripts/train_MLP.py --device cpu --dataclass medium --epochs 3
```

### Extended MLP benchmark

`scripts/train_MLP_exp.py` is the main script for DataLoader, precision, and
GPU-resident-data experiments.

Examples:

```bash
python scripts/train_MLP_exp.py \
  --device cuda \
  --dataclass fast \
  --model big \
  --epochs 10 \
  --profile
```

```bash
python scripts/train_MLP_exp.py \
  --device cuda \
  --dataclass fast \
  --model big \
  --batch_size 8192 \
  --data_on_gpu \
  --manual_batching \
  --tf32 \
  --mixed_precision \
  --result_csv results/mlp_run.csv
```

Available MLP sizes:

- `--model tiny`: small reference MLP, roughly 328K parameters.
- `--model big`: default larger MLP, roughly 17.85M parameters.
- `--model huge`: very large MLP for compute-bound precision experiments.

### 1D ResNet

Single GPU:

```bash
python scripts/train_resnet1d.py \
  --base_width 64 \
  --depth 2 \
  --batch_size 8192 \
  --epochs 10 \
  --tf32 \
  --mixed_precision
```

Single-node multi-GPU:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_resnet1d.py \
  --base_width 64 \
  --depth 2 \
  --batch_size 8192 \
  --epochs 10 \
  --tf32 \
  --mixed_precision \
  --result_csv results/resnet1d.csv
```

### Dual CNN + Transformer model

Single GPU:

```bash
python scripts/train_dual.py \
  --cnn_width 64 \
  --cnn_depth 4 \
  --tf_width 128 \
  --tf_depth 4 \
  --tf_heads 4 \
  --batch_size 8192 \
  --epochs 10 \
  --tf32 \
  --mixed_precision
```

Single-node multi-GPU:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_dual.py \
  --cnn_width 64 \
  --cnn_depth 4 \
  --tf_width 128 \
  --tf_depth 4 \
  --tf_heads 4 \
  --batch_size 8192 \
  --epochs 10 \
  --tf32 \
  --mixed_precision \
  --result_csv results/dual.csv
```

For the DDP scripts, `--batch_size` is the global batch size. Each rank processes
`batch_size / world_size` samples, so the batch size must be divisible by the
number of processes.

## Benchmark Workflows

The `slurm` directory contains runnable templates for the workshop HPC system.
Typical entry points:

```bash
cd slurm
sbatch submit_preprocess.slurm
sbatch 0_submit_MLP.slurm
sbatch benchmark_MLP.slurm
sbatch benchmark_MLP_snakemake.slurm
```

Strong-scaling sweeps use `Snakefile_scaling` inside a single allocated GPU
node. The H100 workflow tests 1, 2, and 4 GPUs. The L40S workflow tests 1, 2,
4, and 8 GPUs.

```bash
cd slurm
sbatch scaling_h100.slurm
sbatch scaling_l40s.slurm
sbatch benchmark_scaling_snakemake.slurm
```

To run the scaling Snakemake workflow manually inside an allocation:

```bash
cd slurm
snakemake -s Snakefile_scaling --cores 1 -p --config node=h100
snakemake -s Snakefile_scaling --cores 1 -p --config node=l40s
```

The scaling workflow writes per-run CSV files under `slurm/results/` and a
combined CSV such as `benchmark_combined_h100.csv` or
`benchmark_combined_l40s.csv`.

## Plotting Results

Benchmark plots can be generated from Python or a notebook.

```python
from src.plot_scaling import load_results, plot_all_scaling

df = load_results("slurm/benchmark_combined_h100.csv")
fig = plot_all_scaling(df)
fig.savefig("scaling_h100.png", dpi=150, bbox_inches="tight")
```

For MLP benchmark CSVs:

```python
from src.plot_benchmark import plot_all

fig = plot_all("slurm/benchmark_combined.csv")
fig.savefig("benchmark_mlp.png", dpi=150, bbox_inches="tight")
```

For resource monitor output:

```python
from src.plot_resources import plot_resources

fig = plot_resources("train_MLP_gpu_h100.csv")
fig.savefig("resources.png", dpi=150, bbox_inches="tight")
```

## Project Layout

```text
.
|-- benchmarks/          # small throughput benchmark scripts
|-- data/                # raw and processed ECG data
|-- notebooks/           # exploratory notebooks
|-- outputs/             # benchmark outputs
|-- scripts/             # download, preprocess, train, monitor scripts
|-- slurm/               # SLURM and Snakemake workflow files
|-- src/                 # datasets, models, plotting utilities
|-- HPC_pytorch_AI.toml  # HPC environment description
|-- Makefile             # convenience targets
|-- modules_for_ecg_ai   # HPC module loads
`-- Pytorch.pdf          # workshop slides
```


## License

This repository is licensed under the GPL-3.0 license. See `LICENSE`.
