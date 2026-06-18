#!/bin/bash -l
#SBATCH -J AI_profile
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:10:00
#SBATCH -p gpu_h100
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH -o train_MLP_profile_%j.out

source /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate

# Output dir for the torch.profiler trace -- unique per job so repeated
# runs don't overwrite each other. SLURM_JOB_ID is set automatically.
TRACE_DIR="torch_profiler_logs_${SLURM_JOB_ID}"

# A handful of profiled steps is enough -- traces grow fast and profiling
# adds CPU/GPU overhead, so don't profile a full training run with this.
python ../scripts/train_MLP_torch_profiler.py \
    --epochs 1 \
    --device cuda \
    --dataclass fast \
    --torch_profile \
    --torch_profile_dir "${TRACE_DIR}" \
    --torch_profile_steps 10 &
TRAIN_PID=$!

python ../scripts/resource_monitor.py \
    --interval 1 \
    --out "train_MLP_profile_${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}.csv" \
    --pid ${TRAIN_PID} &
MONITOR_PID=$!

# Wait for training to finish first, THEN stop the monitor -- backgrounding
# both and only waiting on TRAIN_PID (as in the original script) risks the
# job ending while the monitor is still mid-write.
wait $TRAIN_PID
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo "Trace written to: ${TRACE_DIR}"
echo "Copy this directory back to your laptop or login node to view with TensorBoard."

