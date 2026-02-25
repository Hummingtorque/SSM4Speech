#!/bin/bash
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=gsc_gate_probe_mean
#SBATCH --output=/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2/slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=wluee98
#SBATCH --partition=spgpu
#SBATCH --gpu_cmode=shared

set -euo pipefail

WORKDIR="/nfs/turbo/coe-wluee/hmingtao/SSMnoJax/event-ssm_ver2"
RUN_DIR="${WORKDIR}/outputs/42942634_gsc_train_mamba"

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements_minimal.txt

export PYTHONUNBUFFERED=1

python -u experiment_gate_mlp_probe.py \
  --run-dir "${RUN_DIR}" \
  --max-samples 4096 \
  --batch-size 64 \
  --probe-epochs 20 \
  --probe-lr 1e-3 \
  --feature-mode mean_gate \
  --seed 1234 \
  --out-json "${RUN_DIR}/gate_mlp_probe_meangate_result_${SLURM_JOB_ID}.json"
