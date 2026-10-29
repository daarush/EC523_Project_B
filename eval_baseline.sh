#!/bin/bash -l
#$ -P ec523kb
#$ -N fsn_eval
#$ -l h_rt=02:00:00
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -pe omp 4
#$ -j y
#$ -o logs/

module purge
module load miniconda
conda activate projb

PROJ=/projectnb/ec523kb/students/aarush/project_b
RUN=$PROJ/runs/baseline
cd "$PROJ"
echo "=== node / GPU ==="; hostname; nvidia-smi -L; date
grep -c padded_len cli/submission/model.py | sed 's/^/padded_len occurrences in model.py (must be >0): /'

echo "=== edge cases + input-level test ==="
python - <<'EOF'
import sys, glob, torch, numpy as np, soundfile as sf
sys.path.insert(0, 'cli')
from submission.model import build_enhancer
from safetensors.torch import load_file
m = build_enhancer(16000); m.load_state_dict(load_file('runs/baseline/best.safetensors')); m.eval().cuda()
for T in [1, 100, 255, 256, 257, 511, 512, 513, 4000, 16000*30]:
    x = (torch.randn(1, T)*0.05).cuda(); y = m.enhance(x, 16000)
    ok = y.shape == x.shape and torch.isfinite(y).all().item() and y.abs().max().item() <= 1.0
    print('len', T, tuple(y.shape), 'OK' if ok else 'BAD', round(y.abs().max().item(), 3))
y = m.enhance(torch.zeros(1, 5000).cuda(), 16000); print('silence finite:', torch.isfinite(y).all().item())
x, _ = sf.read(sorted(glob.glob('runs/baseline/dev/noisy/*.wav'))[0], dtype='float32')
for pk in [0.001, 0.1, 0.5, 0.9, 0.99, 1.0]:
    y = m.enhance(torch.from_numpy(x/np.abs(x).max()*pk)[None].cuda(), 16000)
    print('input peak', pk, '-> output peak', round(y.abs().max().item(), 3),
          'OK' if y.abs().max().item() <= 1.0 else 'REJECTED')
EOF

echo "=== resources (final model.py) ==="
python -u scripts/measure_resources.py --out "$RUN/resources.json" || echo "RESOURCE MEASUREMENT FAILED"

echo "=== official CLI on full dev split ==="
cd "$PROJ/cli"
CK=$RUN/best.safetensors
rm -rf "$RUN/pred_noisy" "$RUN/pred_clean"       # remove partial outputs from the failed run
{
  python enhance.py --checkpoint "$CK" --input "$RUN/dev/noisy" --output "$RUN/pred_noisy" --device cuda || echo "enhance(noisy) FAILED"
  python evaluate.py --clean "$RUN/dev/clean" --enhanced "$RUN/pred_noisy" --report "$RUN/noisy_report.json" || echo "evaluate(noisy) FAILED"
  python enhance.py --checkpoint "$CK" --input "$RUN/dev/clean" --output "$RUN/pred_clean" --device cuda || echo "enhance(clean) FAILED"
  python evaluate.py --clean "$RUN/dev/clean" --enhanced "$RUN/pred_clean" --report "$RUN/clean_preservation_report.json" || echo "evaluate(clean) FAILED"
  python score.py --noisy-report "$RUN/noisy_report.json" --clean-report "$RUN/clean_preservation_report.json" || echo "score FAILED"
  python evaluate.py --clean "$RUN/dev/clean" --enhanced "$RUN/dev/noisy" --report "$RUN/noisy_input_report.json" || echo "evaluate(noisy input) FAILED"
} > "$RUN/cli_scores.txt" 2>&1
echo "=== done ==="; date