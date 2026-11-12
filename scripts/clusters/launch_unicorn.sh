#!/usr/bin/env bash
# Run a launcher as a SLURM job on the unicorn cluster (partition aimi). Submit from the repo root;
# job logs go to logs/. The defaults below fit training on 2 GPUs, and sbatch flags override them.
#
# Usage:
#   sbatch [sbatch flags] scripts/clusters/launch_unicorn.sh <launcher> <launcher args>...
#
# Examples:
#   sbatch scripts/clusters/launch_unicorn.sh \
#       scripts/skyrl/launch_skyrl.sh train e5 data/pw/depth_16_size_1000_seed_1_50k
#   sbatch --gres=gpu:1 --ntasks=16 --mem=128GB --time=06:00:00 scripts/clusters/launch_unicorn.sh \
#       scripts/skyrl/launch_skyrl.sh eval qwen3-4b data/hp500 --out_dir=out__eval=hp500

# Defaults: training on 2 GPUs
#SBATCH --job-name=phantom-envs
#SBATCH --partition=aimi
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --gres=gpu:2
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

# sbatch runs a copy of this script from the spool dir, so go back to where it was submitted.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}"
exec "$@"
