#!/bin/bash
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=gsc_train_jetson35
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=00-24:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
GSC_DIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/data/GSC"
RESUME_CKPT=""

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

# Jetson-aligned environment lockfile
python -m pip install --user --upgrade pip
REQ_FILE="requirements_minimal.txt"
if [[ -f /etc/nv_tegra_release ]]; then
  # On Jetson devices, use the full hardware-aligned lockfile.
  REQ_FILE="requirements.txt"
fi
echo "[*] Using dependency file: ${REQ_FILE}"
python -m pip install --user --no-cache-dir -r "${REQ_FILE}"

export PYTHONUNBUFFERED=1
RUN_TAG="${SLURM_JOB_ID}_${SLURM_JOB_NAME}"

CMD=(python -u run_training.py data_dir="${GSC_DIR}" \
  +training.root="${GSC_DIR}" \
  ++output_dir="./outputs/${RUN_TAG}" \
  task.name=gsc-mel-classification model=GSC/medium \
  training.num_epochs=30 training.validate_on_test=false \
  training.per_device_batch_size=64 training.per_device_eval_batch_size=64 \
  training.num_workers=0 \
  training.cut_mix=0.0 \
  +training.mel_bins=64 \
  model.ssm.classification_mode=attnpool model.ssm.pooling_mode=timepool model.ssm.pooling_stride=4 \
  +model.audio_encoder.use=true +model.audio_encoder.kernel_size=3 +model.audio_encoder.stride=1 \
  +model.ssm.input_gate=true +model.ssm.input_gate_rank=0 \
  +model.ssm.input_gate_mode=energy_aware +model.ssm.input_gate_energy_scale_init=0.5 +model.ssm.input_gate_bias_init=0.5 +model.ssm.input_gate_min=0.05 \
  +training.use_log_mel=true \
  +training.specaugment_prob=0.3 +training.freq_mask_param=6 +training.time_mask_param=15 +training.num_masks=1 \
  +training.mixup_alpha=0.1 +training.mixup_prob=0.2 \
  optimizer.ssm_base_lr=2e-05 optimizer.lr_factor=4 optimizer.warmup_epochs=10 ++optimizer.accumulation_steps=1 \
  +optimizer.label_smoothing=0.05 +optimizer.grad_clip_norm=0.5 \
  +optimizer.gate_reg_weight=0.05 +optimizer.gate_reg_margin=0.03 +optimizer.gate_reg_bg_batch_size=8 +optimizer.gate_reg_interval=4 +optimizer.gate_reg_bg_dir="${GSC_DIR}/_background_noise_" \
  optimizer.ssm_weight_decay=0.0 optimizer.weight_decay=0.02)

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(+training.from_checkpoint="${RESUME_CKPT}")
fi

"${CMD[@]}"
