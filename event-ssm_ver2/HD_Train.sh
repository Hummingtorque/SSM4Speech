#!/bin/bash
# python
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=hd_train
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/hd_train_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=00-10:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
HD_DIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/data/HD"
RESUME_CKPT=""

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

# Dependencies
python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements.txt
python -m pip install --user snntorch

export PYTHONUNBUFFERED=1

CMD=(python -u run_training.py data_dir="${HD_DIR}" \
  +training.root="${HD_DIR}/hd_audio" \
  task.name=hd-audio-mel-classification model=shd/medium \
  training.num_epochs=100 training.validate_on_test=false \
  training.per_device_batch_size=64 training.per_device_eval_batch_size=64 \
  training.cut_mix=0.0 \
  +training.mel_bins=64 \
  model.ssm.classification_mode=timepool model.ssm.pooling_mode=timepool model.ssm.pooling_stride=4 \
  model.ssm.dropout=0.15 \
  +model.audio_encoder.use=true +model.audio_encoder.kernel_size=3 +model.audio_encoder.stride=1 \
  +training.use_log_mel=true \
  optimizer.ssm_base_lr=2e-05 optimizer.lr_factor=4 optimizer.warmup_epochs=10 \
  +optimizer.label_smoothing=0.05 +optimizer.grad_clip_norm=0.5 \
  optimizer.ssm_weight_decay=0.0 optimizer.weight_decay=0.02)

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(+training.from_checkpoint="${RESUME_CKPT}")
fi

"${CMD[@]}"


