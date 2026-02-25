#!/bin/bash
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=gsc_gateabl50
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=01-12:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
GSC_DIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/data/GSC"
RESUME_CKPT=""

# Passed via sbatch --export=ALL,RUN_SEED=...,USE_GATE=true/false
RUN_SEED="${RUN_SEED:-1234}"
USE_GATE="${USE_GATE:-true}"

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements_minimal.txt

export PYTHONUNBUFFERED=1
RUN_TAG="${SLURM_JOB_ID}_${SLURM_JOB_NAME}_s${RUN_SEED}"

CMD=(python -u run_training.py data_dir="${GSC_DIR}" \
  seed="${RUN_SEED}" \
  +training.root="${GSC_DIR}" \
  ++output_dir="./outputs/${RUN_TAG}" \
  task.name=gsc-mel-classification model=GSC/medium \
  training.num_epochs=50 training.validate_on_test=false \
  training.per_device_batch_size=64 training.per_device_eval_batch_size=64 \
  training.num_workers=0 \
  training.cut_mix=0.0 \
  +training.mel_bins=64 \
  model.ssm.classification_mode=timepool model.ssm.pooling_mode=timepool model.ssm.pooling_stride=4 \
  +model.audio_encoder.use=true +model.audio_encoder.kernel_size=3 +model.audio_encoder.stride=1 \
  +model.ssm.input_gate="${USE_GATE}" +model.ssm.input_gate_rank=0 +model.ssm.input_gate_mode=sigmoid +model.ssm.input_gate_min=0.0 \
  +training.use_log_mel=true \
  +training.specaugment_prob=0.0 +training.freq_mask_param=0 +training.time_mask_param=0 +training.num_masks=0 \
  +training.mixup_alpha=0.0 +training.mixup_prob=0.0 \
  optimizer.ssm_base_lr=2e-05 optimizer.lr_factor=4 optimizer.warmup_epochs=10 ++optimizer.accumulation_steps=1 \
  +optimizer.label_smoothing=0.05 +optimizer.grad_clip_norm=0.5 \
  +optimizer.gate_reg_weight=0.0 \
  optimizer.ssm_weight_decay=0.0 optimizer.weight_decay=0.02)

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(+training.from_checkpoint="${RESUME_CKPT}")
fi

"${CMD[@]}"
