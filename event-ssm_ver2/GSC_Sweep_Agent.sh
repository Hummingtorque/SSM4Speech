#!/bin/bash
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=gsc_sweep_agent
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01-12:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

# Keep runtime env consistent with the training scripts.
python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements_minimal.txt

export PYTHONUNBUFFERED=1

# Default sweep from latest creation (can be overridden by arg1).
# Usage:
#   sbatch GSC_Sweep_Agent.sh
#   sbatch GSC_Sweep_Agent.sh hmingtao-university-of-michigan/SSMnoJax-event-ssm_ver2/5htlfnls
DEFAULT_SWEEP_PATH="hmingtao-university-of-michigan/SSMnoJax-event-ssm_ver2/5htlfnls"
SWEEP_PATH="${1:-${DEFAULT_SWEEP_PATH}}"

python -m wandb agent "${SWEEP_PATH}"
