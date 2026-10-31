#!/bin/bash -l
#$ -P ec523kb
#$ -N fsn_train
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 4
#$ -j y
#$ -cwd
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to the course 16khz directory}"
: "${TRAIN_MANIFEST:?Set TRAIN_MANIFEST to the approved training file list}"
: "${DEV_MANIFEST:?Set DEV_MANIFEST to the approved development file list}"
# Submit from the repository root. Configure the Python environment before this script.
PROJ="${PROJECT_DIR:-${SGE_O_WORKDIR:-$PWD}}"
cd "$PROJ"
RUN="${RUN_DIR:-runs/baseline_reproduction}"
python -u scripts/train.py --data-root "$DATA_ROOT" \
  --train-manifest "$TRAIN_MANIFEST" --dev-manifest "$DEV_MANIFEST" \
  --out "$RUN" --epochs "${EPOCHS:-30}" --batch-size 8 --workers 3 \
  --device cuda --clean-loss-weight "${CLEAN_LOSS_WEIGHT:-0}"
python -u scripts/measure_resources.py --out "$RUN/resources.json" --clean-loss-weight "${CLEAN_LOSS_WEIGHT:-0}"
python -u scripts/validate_checkpoint.py --checkpoint "$RUN/best.safetensors" \
  --device cuda --out "$RUN/validation.json"
python -u scripts/evaluate_run.py --checkpoint "$RUN/best.safetensors" \
  --dev-dir "$RUN/dev" --out "$RUN/evaluation" --device cuda
