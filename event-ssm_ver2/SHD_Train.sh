#!/bin/bash
# python
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=shd_train
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/train_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=00-8:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
DATA_DIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/data"
# Optional: set to an absolute checkpoint dir to resume training
RESUME_CKPT=""

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

# Optional short test run toggle: set TEST_RUN=1 when submitting to reduce runtime
TEST_RUN="${TEST_RUN:-0}"

# Environment: install pinned dependencies to user site (no JAX)
python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements.txt

export PYTHONUNBUFFERED=1

CMD=(python -u run_training.py data_dir="${DATA_DIR}" model=shd/medium \
  model.ssm.dropout=0.15 \
  training.num_epochs=180 training.validate_on_test=false \
  optimizer.ssm_base_lr=3e-05 optimizer.lr_factor=8 optimizer.warmup_epochs=10 \
  +optimizer.label_smoothing=0.05 +optimizer.grad_clip_norm=1.0 \
  optimizer.ssm_weight_decay=0.0 optimizer.weight_decay=0.02)

# If running a quick test, switch to tiny model and minimal epochs/batch sizes
if [[ "${TEST_RUN}" == "1" ]]; then
  echo "[*] TEST_RUN enabled: using tiny model, 1 epoch, small batches"
  CMD+=(model=shd/tiny training.num_epochs=1 training.per_device_batch_size=8 training.per_device_eval_batch_size=8 logging.interval=50)
fi
if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(+training.from_checkpoint="${RESUME_CKPT}")
fi

"${CMD[@]}"


