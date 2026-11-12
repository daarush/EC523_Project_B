#!/bin/bash -l
# Usage (from project_b/):  qsub -N <name> train_mod.sh <name> [extra train.py flags]
#$ -P ec523kb
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 4
#$ -j y
#$ -o logs/
set -uo pipefail

NAME=${1:?first argument must be the run name}; shift
: "${DATA_ROOT:?export DATA_ROOT}"
: "${TRAIN_MANIFEST:?export TRAIN_MANIFEST}"
: "${DEV_MANIFEST:?export DEV_MANIFEST}"

module purge
module load miniconda
conda activate projb

PROJ=/projectnb/ec523kb/students/aarush/project_b
RUN=$PROJ/runs/$NAME
mkdir -p "$RUN"
cd "$PROJ"

echo "=== $NAME | flags: $* ==="; hostname; nvidia-smi -L; date

echo "=== resources ==="
mkdir -p "$PROJ/resources"
python -u scripts/measure_resources.py --out "$PROJ/resources/$NAME.json" || echo "RESOURCE MEASUREMENT FAILED"

RESUME=""
[ -f "$RUN/last.pt" ] && RESUME="--resume"

echo "=== training ==="; date
python -u scripts/train.py --data-root "$DATA_ROOT" \
  --train-manifest "$TRAIN_MANIFEST" --dev-manifest "$DEV_MANIFEST" \
  --out "runs/$NAME" --epochs 30 --batch-size 8 --workers 3 --device cuda \
  --seed 0 $RESUME "$@"
echo "=== training exit code: $? ==="; date

echo "=== official CLI scoring on dev split ==="
cd "$PROJ/cli"
CK=$RUN/best.safetensors
{
  python enhance.py --checkpoint "$CK" --input "$RUN/dev/noisy" --output "$RUN/pred_noisy" --device cuda || echo "enhance(noisy) FAILED"
  python evaluate.py --clean "$RUN/dev/clean" --enhanced "$RUN/pred_noisy" --report "$RUN/noisy_report.json" || echo "evaluate(noisy) FAILED"
  python enhance.py --checkpoint "$CK" --input "$RUN/dev/clean" --output "$RUN/pred_clean" --device cuda || echo "enhance(clean) FAILED"
  python evaluate.py --clean "$RUN/dev/clean" --enhanced "$RUN/pred_clean" --report "$RUN/clean_preservation_report.json" || echo "evaluate(clean) FAILED"
  python score.py --noisy-report "$RUN/noisy_report.json" --clean-report "$RUN/clean_preservation_report.json" || echo "score FAILED"
  python evaluate.py --clean "$RUN/dev/clean" --enhanced "$RUN/dev/noisy" --report "$RUN/noisy_input_report.json" || echo "evaluate(noisy input) FAILED"
} > "$RUN/cli_scores.txt" 2>&1
echo "=== done ==="; date