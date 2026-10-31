#!/bin/bash -l
#$ -P ec523kb
#$ -N fsn_eval
#$ -l h_rt=02:00:00
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 4
#$ -j y
#$ -cwd
set -euo pipefail
: "${CHECKPOINT:?Set CHECKPOINT to the nominated safetensors file}"
: "${DEV_DIR:?Set DEV_DIR to a directory containing clean/ and noisy/}"
: "${EVAL_OUT:?Set EVAL_OUT to a new output directory}"
PROJ="${PROJECT_DIR:-${SGE_O_WORKDIR:-$PWD}}"
cd "$PROJ"
python scripts/validate_checkpoint.py --checkpoint "$CHECKPOINT" \
  --device cuda --out "${EVAL_OUT}_validation.json"
python scripts/evaluate_run.py --checkpoint "$CHECKPOINT" \
  --dev-dir "$DEV_DIR" --out "$EVAL_OUT" --device cuda
