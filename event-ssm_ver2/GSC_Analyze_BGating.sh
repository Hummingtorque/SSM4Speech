#!/bin/bash
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=gsc_analyze_bgating
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
RUN_DIR="${WORKDIR}/outputs/42557048_gsc_train_mamba"
BG_DIR="${WORKDIR}/data/GSC/_background_noise_"
OUT_JSON="${RUN_DIR}/gating_keyword_vs_background_stats_${SLURM_JOB_ID}.json"

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements.txt

export PYTHONUNBUFFERED=1

python -u analyze_b_gating.py \
  --run-dir "${RUN_DIR}" \
  --background-dir "${BG_DIR}" \
  --max-keyword-samples 4096 \
  --background-samples 4096 \
  --background-batch-size 64 \
  --background-rounds 8 \
  --seed 1234 \
  --eval-batch-size 16 \
  --analysis-pad-unit 1024 \
  --out-json "${OUT_JSON}"

echo "[*] Saved gating stats to: ${OUT_JSON}"
