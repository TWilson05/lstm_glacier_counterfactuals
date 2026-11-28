#!/bin/bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo "Usage: bash hpc/submit_partial_glacier.sh <experiment_name> <model_type>" >&2
    exit 1
fi
EXP_NAME=$1
MODEL_TYPE=$2
if [ ! -f secrets.env ]; then
    echo "Error: secrets.env must contain EMAIL and ACCOUNT" >&2
    exit 1
fi
source secrets.env
mkdir -p logs
sbatch --account="$ACCOUNT" --mail-user="$EMAIL" \
    --export="ALL,EXP_NAME=$EXP_NAME,MODEL_TYPE=$MODEL_TYPE" hpc/partial_glacier_job.sh
