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


mpirun -np $SLURM_NTASKS python ../scripts/train_MLP.py --epochs 10 --device cuda --dataclass fast --profile
#mpirun -np $SLURM_NTASKS nsys profile python laplacian_gpu.py
