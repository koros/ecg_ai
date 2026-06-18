#!/bin/bash -l
#SBATCH -J AI
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:10:00
#SBATCH -p gpu_h100
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#####SBATCH -o DNA_out_100000000_N1

source  /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate


python ../scripts/train_MLP.py --epochs 100 --device cuda --dataclass fast --profile &
TRAIN_PID=$!
python ../scripts/resource_monitor.py --interval 1 --out train_MLP_${SLURM_JOB_PARTITION}.csv --pid ${TRAIN_PID} &
MONITOR_PID=$!
wait $TRAIN_PID
kill $MONITOR_PID 2>/dev/null   # then stop the monitor cleanly
wait $MONITOR_PID 2>/dev/null   # let it flush/exit

#mpirun -np $SLURM_NTASKS nsys profile python laplacian_gpu.py
