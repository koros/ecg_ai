#!/bin/bash -l
#SBATCH -J scaling_l40s
#SBATCH -N 1
#SBATCH -n 1                       # one task; torchrun launches per-GPU processes
#SBATCH -c 8                       # cores per task -- 8 per GPU on a 64-core/8-GPU node
#SBATCH -t 02:00:00
#SBATCH -p gpu_l40s
#SBATCH --gres=gpu:8               # all 8 L40s on the node
#SBATCH -o scaling_l40s_%j.out

# Same structure as scaling_h100.slurm; the partition and per-task core
# count differ. The same train scripts are used -- DDP behavior is
# identical; what differs is NVLink (H100) vs PCIe (L40s) interconnect,
# which is exactly the interesting variable to compare across the two
# nodes when looking at scaling efficiency.

source /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate

SCRIPT="${SCRIPT:-../scripts/train_resnet1d.py}"
NPROC="${NPROC:-8}"                # number of GPUs to use (1, 2, 4, or 8)
EXTRA_ARGS="${EXTRA_ARGS:---base_width 64 --depth 2}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
RESULT_CSV="${RESULT_CSV:-results_l40s.csv}"

echo "Running ${SCRIPT} with NPROC=${NPROC} ${EXTRA_ARGS}"

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    ${SCRIPT} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --tf32 --mixed_precision \
    --result_csv ${RESULT_CSV} \
    ${EXTRA_ARGS}
