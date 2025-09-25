#!/bin/bash
#SBATCH -J testslurm
#SBATCH -o ios/sac-GRU-online.out 
#SBATCH -e ios/sac-GRU-online.err 
#SBATCH -N 1 
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:a800:1 
#SBATCH --mem=32G
#SBATCH--cpus-per-task=16

export http_proxy=http://10.1.27.7:17891
export https_proxy=http://10.1.27.7:17891
export HTTP_PROXY=http://10.1.27.7:17891
export HTTPS_PROXY=http://10.1.27.7:17891

export LD_LIBRARY_PATH=/data/home/name/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PATH="$LD_LIBRARY_PATH:$PATH"

# export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

. $HOME/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate sac
wandb login 6330db7b5b23192bb0722c1f5499fc8d607e7221
wandb online
python3 sac_continuous_action_multi_steps_GRU_jax.py