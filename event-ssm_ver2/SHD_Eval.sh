#!/bin/bash
#python
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=shd_eval
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/eval_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00-00:30:00
#SBATCH --account=wluee0
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
CHECKPOINT_DIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/outputs/2025-11-05-00-48-35/checkpoints"

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

export PYTHONUNBUFFERED=1
python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements.txt

python -u run_evaluation.py task=spiking-heidelberg-digits model.ssm.num_layers_per_stage=3 training.per_device_eval_batch_size=4 training.num_workers=0 data_dir=/nfs/turbo/coe-wluee/hmingtao/SSM_SHD/event-ssm_ver2/data checkpoint="${CHECKPOINT_DIR}"