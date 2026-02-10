#!/bin/bash
# python
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=gsc_inc_sota
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/gsc_inc_sota_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=02-12:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
GSC_DIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/data/GSC"
RESUME_CKPT=""
INCLUDE_LABELS=""

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

# Dependencies
python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements.txt
python -m pip install --user snntorch

export PYTHONUNBUFFERED=1

# SOTA Configuration: Inception2d + SpecAugment + Mixup + Long Training
CMD=(python -u run_training.py data_dir="${GSC_DIR}" \
  +training.root="${GSC_DIR}" \
  task.name=gsc-mel-classification model=GSC/medium \
  training.num_epochs=200 training.validate_on_test=false \
  training.per_device_batch_size=32 training.per_device_eval_batch_size=64 \
  training.num_workers=4 \
  training.cut_mix=0.0 \
  +training.mel_bins=64 \
  model.ssm.classification_mode=timepool model.ssm.pooling_mode=timepool model.ssm.pooling_stride=4 \
  +model.audio_encoder.use=true +model.audio_encoder.type=inception2d +model.audio_encoder.in_channels=1 \
  +model.audio_encoder.kernel_size=3 +model.audio_encoder.stride=1 +model.audio_encoder.freq_stride=2 \
  +training.use_log_mel=true \
  +training.specaugment_prob=0.3 +training.freq_mask_param=8 +training.time_mask_param=15 +training.num_masks=1 \
  +training.mixup_alpha=0.1 +training.mixup_prob=0.2 \
  optimizer.ssm_base_lr=2e-05 optimizer.lr_factor=4 optimizer.warmup_epochs=15 ++optimizer.accumulation_steps=2 \
  +optimizer.label_smoothing=0.02 +optimizer.grad_clip_norm=0.5 \
  optimizer.ssm_weight_decay=0.0 optimizer.weight_decay=0.02)

if [[ -n "${INCLUDE_LABELS}" ]]; then
  CMD+=(+training.include_labels="${INCLUDE_LABELS}")
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(+training.from_checkpoint="${RESUME_CKPT}")
fi

"${CMD[@]}"

