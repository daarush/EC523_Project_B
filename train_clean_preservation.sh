#!/bin/bash
#$ -P ec523kb
#$ -N fsn_cleanloss
#$ -cwd
#$ -j y
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l h_rt=12:00:00
#$ -pe omp 4
source /etc/profile
set -eo pipefail
module load python3/3.12.4
: "${VENV_PATH:?Export VENV_PATH and pass it with qsub -v VENV_PATH}"
source "$VENV_PATH/bin/activate"
export DATA_ROOT="${DATA_ROOT:-/projectnb/ec523kb/projects/DeNoise/16khz}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m unittest discover -s tests -v
python -u scripts/run_midpoint.py --run "${RUN_DIR:-runs/clean_preservation_reproduction}"
