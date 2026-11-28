#!/bin/bash
#SBATCH --job-name=glacier_removal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --array=0-9%5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/removal_%A_%a.out
#SBATCH --error=logs/removal_%A_%a.err

set -euo pipefail
module purge
module load intel-oneapi-compilers/2023.1.0
module load python/3.11.6
source venv/bin/activate
python -u -m src.partial_glacier_inference --exp_name "$EXP_NAME" --model_type "$MODEL_TYPE" --member_id "$SLURM_ARRAY_TASK_ID"
