#!/bin/bash
#SBATCH --mail-user=hmingtao@umich.edu
#SBATCH --mail-type=END
#SBATCH --job-name=gsc_snr_compare
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
RUN_A="${WORKDIR}/outputs/42477123_gsc_train"
RUN_B="${WORKDIR}/outputs/42557048_gsc_train_mamba"

cd "${WORKDIR}"

module load python/3.11
module load cuda/12.6

python -m pip install --user --upgrade pip
python -m pip install --user --no-cache-dir -r requirements_minimal.txt

export PYTHONUNBUFFERED=1

python -u snr_robustness_compare.py \
  --run-a "${RUN_A}" \
  --run-b "${RUN_B}" \
  --name-a no_gate \
  --name-b with_gate \
  --snrs clean,20,10,5,0,-5 \
  --batch-size 64 \
  --seed 1234 \
  --out-json "${WORKDIR}/outputs/snr_compare_${SLURM_JOB_ID}.json"
