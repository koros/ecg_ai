#!/bin/bash -l
#SBATCH -J AI_benchmark
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -p gpu_h100
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH -o benchmark_%j.out

source /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate

SCRIPT="../scripts/train_MLP_final.py"
RESULTS="benchmark_${SLURM_JOB_ID}.csv"

# CSV header
echo "tier,dataclass,num_workers,pin_memory,tf32,mixed_precision,epoch_time_s,accuracy" > ${RESULTS}

# Helper: run one configuration, extract last epoch time and accuracy,
# append a row to the CSV.
#
# Usage: run_config tier dataclass num_workers pin_memory tf32 mixed_precision
#
run_config() {
    local tier=$1
    local dataclass=$2
    local num_workers=$3
    local pin_memory=$4       # "yes" or "no"
    local tf32=$5             # "yes" or "no"
    local mixed_precision=$6  # "yes" or "no"

    # Build the argument list from the yes/no flags
    local extra_args=""
    [ "$pin_memory"      = "yes" ] && extra_args="$extra_args --pin_memory"
    [ "$tf32"            = "yes" ] && extra_args="$extra_args --tf32"
    [ "$mixed_precision" = "yes" ] && extra_args="$extra_args --mixed_precision"

    echo ""
    echo "================================================================"
    echo "Tier ${tier} | dataclass=${dataclass} workers=${num_workers} pin=${pin_memory} tf32=${tf32} amp=${mixed_precision}"
    echo "================================================================"

    # Run and capture output; tee so we still see it in the .out file
    local output
    output=$(python ${SCRIPT} \
        --device cuda \
        --dataclass ${dataclass} \
        --num_workers ${num_workers} \
        --epochs ${EPOCHS} \
        ${extra_args} 2>&1 | tee /dev/stderr)

    # Extract last epoch time (last line matching "Epoch N loss=... time=...s")
    local epoch_time
    epoch_time=$(echo "$output" | grep "^Epoch" | tail -1 | sed 's/.*time=\([0-9.]*\)s/\1/')

    # Extract accuracy
    local accuracy
    accuracy=$(echo "$output" | grep "^Accuracy:" | sed 's/Accuracy: //')

    # Append to CSV (use "yes"/"no" strings for readability in the CSV)
    echo "${tier},${dataclass},${num_workers},${pin_memory},${tf32},${mixed_precision},${epoch_time},${accuracy}" >> ${RESULTS}
}

# ----------------------------------------------------------------
# TIER 1 -- DataLoader benchmarks
# Goal: isolate the effect of num_workers and pin_memory on the
# load and transfer stages. Precision fixed (no tf32, no AMP) so
# that the GPU compute time is constant across runs and differences
# are attributable to the DataLoader alone.
#
# Combinations: 3 dataclasses x 3 worker counts x 2 pin_memory = 18 runs
# Epochs: 3 (enough to get stable timing, short enough to be fast)
# ----------------------------------------------------------------
EPOCHS=3

for dataclass in medium fast; do
    for num_workers in 0 2 4; do
        for pin_memory in no yes; do
            run_config 1 $dataclass $num_workers $pin_memory no no
        done
    done
done

# ----------------------------------------------------------------
# TIER 2 -- Precision benchmarks
# Goal: show the effect of TF32 and mixed precision (AMP bf16) on
# GPU compute time, now that the DataLoader is no longer the
# bottleneck (num_workers=4, pin_memory=yes, dataclass=fast).
#
# Combinations: 2 tf32 x 2 mixed_precision = 4 runs
# Epochs: 10 (more epochs needed to see stable compute-bound timing)
# ----------------------------------------------------------------
EPOCHS=10

for tf32 in no yes; do
    for mixed_precision in no yes; do
        run_config 2 fast 4 yes $tf32 $mixed_precision
    done
done

echo ""
echo "================================================================"
echo "All runs complete. Results written to ${RESULTS}"
echo "================================================================"
echo ""
echo "Quick summary (epoch times):"
column -t -s',' ${RESULTS}
