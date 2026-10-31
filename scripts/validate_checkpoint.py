"""Finite, shape, peak and repeatability checks for a nominated checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'cli'))
from submission.model import build_enhancer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    model = build_enhancer(16000)
    model.load_state_dict(load_file(str(a.checkpoint)), strict=True)
    model.to(a.device).eval()
    generator = torch.Generator().manual_seed(523)
    cases = [(f'length_{n}', torch.randn(1, n, generator=generator)*.05)
             for n in (1, 100, 255, 256, 257, 511, 512, 513, 16000, 480000)]
    cases.append(('silence', torch.zeros(1, 16000)))
    t = torch.arange(16000)/16000
    for level in (.001, .5, .99, 1.):
        cases.append((f'tone_peak_{level}', (level*torch.sin(2*torch.pi*440*t))[None]))
    rows = []
    with torch.inference_mode():
        for name, x in cases:
            x = x.to(a.device)
            y, repeat = model.enhance(x, 16000), model.enhance(x, 16000)
            peak = float(y.abs().max())
            ok = y.shape == x.shape and bool(torch.isfinite(y).all()) and peak <= 1 and torch.equal(y, repeat)
            rows.append({'case': name, 'pass': ok, 'peak': peak,
                         'repeat_bitwise_equal': torch.equal(y, repeat)})
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {'passed': all(r['pass'] for r in rows) and count <= 15_000_000,
              'trainable_parameters': count, 'cases': rows,
              'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256((ROOT/'cli/submission/model.py').read_bytes()).hexdigest(),
              'scope': 'sampled synthetic edge cases; not proof for all possible inputs', 'device': a.device}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
