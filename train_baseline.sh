#!/bin/bash -l
# Submit from project_b/ :   mkdir -p logs && qsub train_baseline.sh
#$ -P ec523kb
#$ -N fsn_baseline
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=8.9
#$ -pe omp 4
#$ -j y
#$ -o logs/

# gpu_c=8.9 = L40S (the handout's reference GPU). If the job never starts, check `qgpus`
# and fall back to gpu_c=8.6 (A40) -- but then memory/latency numbers are A40, not L40S.

module purge
module load miniconda
conda activate projb

PROJ=/projectnb/ec523kb/students/aarush/project_b
RUN=$PROJ/runs/baseline
mkdir -p "$RUN"
cd "$PROJ"

echo "=== node / GPU ==="; hostname; nvidia-smi -L; date

# 1) resources first, so they exist even if training later dies
echo "=== resources ==="
python -u scripts/measure_resources.py --out "$RUN/resources.json" || echo "RESOURCE MEASUREMENT FAILED"

# 2) training (--resume: re-submitting this script continues from last.pt)
echo "=== training ==="; date
python -u scripts/train.py --out runs/baseline --epochs 30 --batch-size 8 --workers 3 --resume
echo "=== training exit code: $? ==="; date

# 3) official CLI on the dev split (failures here do not affect the checkpoint)
echo "=== official CLI scoring on dev split ==="
cd "$PROJ/cli"
CK=$RUN/best.safetensors
{
  python enhance.py --checkpoint "$CK" --input "$RUN/dev/noisy" --output "$RUN/pred_noisy" --device cuda || echo "enhance(noisy) FAILED"
  python evaluate.py --clean "$RUN/dev/clean" --enhanced "$RUN/pred_noisy" --report "$RUN/noisy_report.json" || echo "evaluate(noisy) FAILED"

  python enhance.py --checkpoint "$CK" --input "$RUN/dev/clean" --output "$RUN/pred_clean" --device cuda || echo "enhance(clean) FAILED"
  python evaluate.py --clean "$RUN/dev/clean" --enhanced "$RUN/pred_clean" --report "$RUN/clean_preservation_report.json" || echo "evaluate(clean) FAILED"

  python score.py --noisy-report "$RUN/noisy_report.json" --clean-report "$RUN/clean_preservation_report.json" || echo "score FAILED"

  # reference row: the unprocessed noisy input scored against clean
  python evaluate.py --clean "$RUN/dev/clean" --enhanced "$RUN/dev/noisy" --report "$RUN/noisy_input_report.json" || echo "evaluate(noisy input) FAILED"
} > "$RUN/cli_scores.txt" 2>&1
echo "=== done ==="; date