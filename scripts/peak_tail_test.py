"""Test: do output-peak spikes come from the istft tail (last < hop samples)?

Put in project_b/scripts/. Run on a GPU:
  python scripts/peak_tail_test.py --ckpt runs/baseline/best.safetensors --dev-dir runs/baseline/dev

For each file, runs the model (a) as-is and (b) with zeros appended at the end so that
every real sample is covered by >= 2 overlapping frames, then crops back to T.
Prints, for files rejected in (a): length, T % hop, where the peak is, and whether it
sits in the tail region [hop*(T//hop), T). Does not modify any file.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
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
    p.add_argument("--dev-dir", required=True)
    p.add_argument("--cli-dir", default=str(Path(__file__).resolve().parents[1] / "cli"))
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()

    sys.path.insert(0, a.cli_dir)
    import submission.model as M
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M.build_enhancer(SR)
    model.load_state_dict(load_file(a.ckpt))
    model.to(dev).eval()
    hop, n_fft = model.hop, model.n_fft

    root = Path(a.dev_dir)
    files = sorted((root / "noisy").glob("*.wav"))
    files = files[:: max(1, len(files) // a.n)][: a.n]

    def run(x, pad):
        T = len(x)
        xt = torch.from_numpy(x)[None].to(dev)
        if pad:
            P = -(-(T + n_fft // 2) // hop) * hop
            xt = F.pad(xt, (0, P - T))
        return model.enhance(xt, SR)[0, :T].cpu().numpy()

    for kind in ("noisy", "clean"):
        stats = {False: [], True: []}
        print(f"\n=== input: {kind} ===")
        for f in files:
            x, _ = sf.read(root / kind / f.name, dtype="float32")
            y, _ = sf.read(root / "clean" / f.name, dtype="float32")
            en0, en1 = run(x, False), run(x, True)
            pk0, pk1 = float(np.abs(en0).max()), float(np.abs(en1).max())
            stats[False].append((pk0, si_sdr(en0, y)))
            stats[True].append((pk1, si_sdr(en1, y)))
            if pk0 > 1.0:
                T = len(x); idx = int(np.abs(en0).argmax()); E = hop * (T // hop)
                print(f"  rejected {f.name}: T={T} T%hop={T % hop} peak@{idx} "
                      f"({'IN TAIL' if idx >= E else 'in body'}; tail starts at {E}) "
                      f"peak {pk0:.2f} -> with padding {pk1:.2f}")
        for name, pad in (("as-is", False), ("padded", True)):
            pk = np.array([q for q, _ in stats[pad]])
            sd = np.mean([s for _, s in stats[pad]])
            print(f"  {name:7s} SI-SDR {sd:6.2f}  peak med {np.median(pk):.2f} max {pk.max():.2f}  "
                  f"REJECTED {(pk > 1.0).sum()}/{len(pk)}")


if __name__ == "__main__":
    main()