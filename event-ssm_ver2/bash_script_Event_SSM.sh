#!/bin/bash

# python
#SBATCH --mail-user=zxygo@umich.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=10-00:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --job-name=test_running
#SBATCH --gpu_cmode=shared
#python run_training.py task=dvs-gesture
#wandb sweep sweep.yaml
#wandb agent zxygo-university-of-michigan/event-ssm_ver2/r02hqxmw
python read_param.py