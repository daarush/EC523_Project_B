"""Diagnose output-peak overshoot of a trained checkpoint, with mask-cap variants.

Put in project_b/scripts/. Run on a GPU (qrsh) :
  python scripts/peak_diag.py --ckpt runs/baseline/best.safetensors \
      --dev-dir runs/baseline/dev --out runs/baseline/peak_diag.json

For each input kind (noisy, clean) and each cap on the decompressed cIRM
(real and imag parts clamped separately to [-cap, cap]; 'none' = the model's default,
~53), reports mean SI-SDR, peak statistics, and how many files would be REJECTED
by the runner (peak > 1.0). Analysis only; it does not change any file.
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

SR = 16_000


def si_sdr(est, ref, eps=1e-8):
    est = est.astype(np.float64) - est.mean()
    ref = ref.astype(np.float64) - ref.mean()
    a = (est @ ref) / (ref @ ref + 1e-12)
    t = a * ref
    r = est - t
    return 10 * np.log10((t @ t + eps) / (r @ r + eps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dev-dir", required=True)           # contains noisy/ and clean/
    p.add_argument("--cli-dir", default=str(Path(__file__).resolve().parents[1] / "cli"))
    p.add_argument("--n", type=int, default=200)          # evenly spaced subset of dev files
    p.add_argument("--caps", default="none,10,5,3,2,1")
    p.add_argument("--out", default="peak_diag.json")
    a = p.parse_args()

    sys.path.insert(0, a.cli_dir)
    import submission.model as M

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M.build_enhancer(SR)
    model.load_state_dict(load_file(a.ckpt))
    model.to(dev).eval()
    default_limit = M.LIMIT

    root = Path(a.dev_dir)
    files = sorted((root / "noisy").glob("*.wav"))
    files = files[:: max(1, len(files) // a.n)][: a.n]
    print(f"{len(files)} dev files, device {dev}", flush=True)

    results = []
    for kind in ("noisy", "clean"):
        data = []
        for f in files:
            x, _ = sf.read(root / kind / f.name, dtype="float32")
            y, _ = sf.read(root / "clean" / f.name, dtype="float32")
            data.append((f.name, x, y))
        for cap in a.caps.split(","):
            M.LIMIT = default_limit if cap == "none" else M.K_MASK * math.tanh(M.C_MASK * float(cap) / 2)
            peaks, ratios, sdrs = [], [], []
            for name, x, y in data:
                en = model.enhance(torch.from_numpy(x)[None].to(dev), SR)[0].cpu().numpy()
                pk = float(np.abs(en).max())
                peaks.append((pk, name))
                ratios.append(pk / max(float(np.abs(x).max()), 1e-9))
                sdrs.append(si_sdr(en, y))
            pk = np.array([q for q, _ in peaks])
            worst = sorted(peaks, reverse=True)[:3]
            row = {"input": kind, "cap": cap, "mask_limit_compressed": M.LIMIT,
                   "mean_SISDR": float(np.mean(sdrs)),
                   "peak_median": float(np.median(pk)), "peak_p95": float(np.percentile(pk, 95)),
                   "peak_max": float(pk.max()),
                   "median_out_over_in_peak": float(np.median(ratios)),
                   "files_over_1.0": int((pk > 1.0).sum()), "n_files": len(pk),
                   "worst": [(n, round(q, 3)) for q, n in worst]}
            results.append(row)
            print(f"{kind:5s} cap={cap:>4s}  SI-SDR {row['mean_SISDR']:6.2f}  "
                  f"peak med {row['peak_median']:.2f} p95 {row['peak_p95']:.2f} max {row['peak_max']:.2f}  "
                  f"REJECTED {row['files_over_1.0']}/{row['n_files']}", flush=True)
    M.LIMIT = default_limit
    Path(a.out).write_text(json.dumps(results, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()