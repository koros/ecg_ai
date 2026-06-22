#!/bin/bash -l
#SBATCH -J AI_benchmark
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:00:00
#SBATCH -p gpu_h100
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH -o benchmark_%j.out

source /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate

SCRIPT="../scripts/train_MLP_exp.py"
RESULTS="benchmark_MLP_${SLURM_JOB_ID}.csv"

# Remove any stale results file so write_result_csv writes a fresh header.
rm -f ${RESULTS}

# ----------------------------------------------------------------
# TIER 1 -- DataLoader benchmarks
# Goal: isolate the effect of num_workers and pin_memory on the
# load and transfer stages. Precision fixed (no tf32, no AMP),
# data_on_gpu off. "slow" dataclass excluded entirely -- far too
# slow to be worth benchmarking; "medium" included as a realistic
# middle ground against "fast".
#
# Combinations: 2 dataclasses x 3 worker counts x 2 pin_memory = 12 runs
# ----------------------------------------------------------------
echo "=== TIER 1: DataLoader benchmarks ==="
for dataclass in medium fast; do
    for num_workers in 0 2 4; do
        for pin_memory in "" "--pin_memory"; do
            echo "tier1 | dataclass=${dataclass} workers=${num_workers} pin=${pin_memory:-no}"
            python ${SCRIPT} \
                --device cuda \
                --dataclass ${dataclass} \
                --num_workers ${num_workers} \
                ${pin_memory} \
                --epochs 3 \
                --result_csv ${RESULTS}
        done
    done
done

# ----------------------------------------------------------------
# TIER 2 -- Precision benchmarks
# Goal: show the effect of TF32 and mixed precision (AMP bf16) on
# GPU compute time, with the DataLoader already tuned (num_workers=4,
# pin_memory, dataclass=fast) so it's not the bottleneck.
#
# Combinations: 2 tf32 x 2 mixed_precision = 4 runs
# ----------------------------------------------------------------
echo ""
echo "=== TIER 2: Precision benchmarks ==="
for tf32 in "" "--tf32"; do
    for mixed_precision in "" "--mixed_precision"; do
        echo "tier2 | tf32=${tf32:-no} amp=${mixed_precision:-no}"
        python ${SCRIPT} \
            --device cuda \
            --dataclass fast \
            --num_workers 4 \
            --pin_memory \
            ${tf32} \
            ${mixed_precision} \
            --epochs 10 \
            --result_csv ${RESULTS}
    done
done

# ----------------------------------------------------------------
# TIER 3 -- GPU-resident dataset benchmark
# Goal: show the effect of --data_on_gpu against the best DataLoader
# settings from Tier 1, and also against the worst-case (num_workers=0,
# no pin_memory) to show data_on_gpu alone can substitute for DataLoader
# tuning entirely. dataclass fixed to "fast" since --data_on_gpu requires
# an in-memory tensor dataset (fast or medium, not slow).
#
# Combinations: 2 worker/pin configs x 2 data_on_gpu = 4 runs
# ----------------------------------------------------------------
echo ""
echo "=== TIER 3: GPU-resident dataset benchmark ==="
for config in "0 " "4 --pin_memory"; do
    read -r num_workers pin_memory <<< "$config"
    for data_on_gpu in "" "--data_on_gpu"; do
        echo "tier3 | workers=${num_workers} pin=${pin_memory:-no} data_on_gpu=${data_on_gpu:-no}"
        python ${SCRIPT} \
            --device cuda \
            --dataclass fast \
            --num_workers ${num_workers} \
            ${pin_memory} \
            ${data_on_gpu} \
            --epochs 10 \
            --result_csv ${RESULTS}
    done
done

echo ""
echo "================================================================"
echo "All runs complete. Results written to ${RESULTS}"
echo "================================================================"
echo ""
echo "Quick summary:"
column -t -s',' ${RESULTS}
