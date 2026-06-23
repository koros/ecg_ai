#!/bin/bash -l
#SBATCH -J AI
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:00:00
#SBATCH -p gpu_h100
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1

source  /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate

EXE=train_MLP_exp.py
python ../scripts/${EXE} --epochs 10 --device cuda --dataclass fast --profile &

TRAIN_PID=$!
python ../scripts/resource_monitor.py --interval 1 --out train_MLP_${SLURM_JOB_PARTITION}.csv --pid ${TRAIN_PID} &
MONITOR_PID=$!
wait $TRAIN_PID
kill $MONITOR_PID 2>/dev/null   # then stop the monitor cleanly
wait $MONITOR_PID 2>/dev/null   # let it flush/exit


























