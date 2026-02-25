#!/bin/bash
# python
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=gsc_finetune_40779923
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/gsc_finetune_40779923_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=00-12:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
GSC_DIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/data/GSC"
RESUME_CKPT="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/outputs/2026-01-23-19-18-33/checkpoints/model.pt"
# Optional: restrict training/eval to a subset of label folders under ${GSC_DIR}.
# IMPORTANT: quote strings like 'yes'/'no'/'on'/'off' to avoid YAML boolean parsing.
# Example (10 labels):
# INCLUDE_LABELS="['yes','no','up','down','left','right','on','off','stop','go']"
INCLUDE_LABELS=""

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

# Dependencies (snntorch required for GSC spike conversion)
python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements.txt
python -m pip install --user snntorch

export PYTHONUNBUFFERED=1

CMD=(python -u run_training.py data_dir="${GSC_DIR}" \
  +training.root="${GSC_DIR}" \
  task.name=gsc-mel-classification model=GSC/medium \
  training.num_epochs=20 training.validate_on_test=false \
  training.per_device_batch_size=64 training.per_device_eval_batch_size=64 \
  training.num_workers=0 \
  training.cut_mix=0.0 \
  +training.mel_bins=64 \
  model.ssm.classification_mode=timepool model.ssm.pooling_mode=timepool model.ssm.pooling_stride=4 \
  +model.audio_encoder.use=true +model.audio_encoder.kernel_size=3 +model.audio_encoder.stride=1 \
  +model.ssm.input_gate=false +model.ssm.input_gate_rank=0 \
  +training.use_log_mel=true \
  +training.specaugment_prob=0.0 +training.freq_mask_param=0 +training.time_mask_param=0 +training.num_masks=0 \
  +training.mixup_alpha=0.0 +training.mixup_prob=0.0 \
  +training.vad_threshold=0.0 \
  +training.boost_labels="['two','four','go','up']" +training.boost_factor=3 \
  optimizer.ssm_base_lr=1e-05 optimizer.lr_factor=4 optimizer.warmup_epochs=10 ++optimizer.accumulation_steps=1 \
  +optimizer.label_smoothing=0.0 +optimizer.grad_clip_norm=0.5 \
  optimizer.ssm_weight_decay=0.0 optimizer.weight_decay=0.02)

if [[ -n "${INCLUDE_LABELS}" ]]; then
  CMD+=(+training.include_labels="${INCLUDE_LABELS}")
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(+training.from_checkpoint="${RESUME_CKPT}")
fi

"${CMD[@]}"
