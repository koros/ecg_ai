#!/bin/bash -l
#SBATCH -J AI_benchmark
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 12:30:00
#SBATCH -p gpu_h100
#SBATCH --mincpus=4
#SBATCH --gres=gpu:1
#SBATCH -o benchmark_MLP_snakemake_%j.out

source /users/${USER}/ecg_ai/modules_for_ecg_ai
. /users/${USER}/.jupyter_virtualenvs/HPC_pytorch_AI/bin/activate
# dry run : snakemake -n
# run it (not we do not want to run mutliple benchmark at the same time because they will all compete for the same GPU card!
snakemake --cores 1 -p --delete-all-output   # step 1: clean
snakemake --cores 1 -p
