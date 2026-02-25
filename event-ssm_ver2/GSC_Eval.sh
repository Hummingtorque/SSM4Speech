#!/bin/bash
# python
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=gsc_eval
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00-02:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
CFG_DIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/outputs/2026-02-11-12-53-29"
CKPT="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/outputs/2026-02-11-12-53-29/checkpoints/model.pt"
GSC_DIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/data/GSC"

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

# Dependencies
python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements.txt

export PYTHONUNBUFFERED=1

python -u run_evaluation.py \
  --config-path "${CFG_DIR}" --config-name config \
  checkpoint="${CKPT}" \
  data_dir="${GSC_DIR}" \
  ++training.root="${GSC_DIR}" \
  training.per_device_eval_batch_size=16 training.num_workers=0 \
  +training.specaugment_prob=0.0 +training.freq_mask_param=0 +training.time_mask_param=0 +training.num_masks=0 \
  +training.mixup_alpha=0.0 +training.mixup_prob=0.0
