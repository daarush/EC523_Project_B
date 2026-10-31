"""Exercise both objectives and checkpoint export on synthetic tones, NOT speech results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); out = a.out.resolve()
    if out.exists():
        raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True)
    for part in ('clean', 'noisy'):
        (out/'data'/f'{part}_trainset_28spk_wav').mkdir(parents=True)
    rng = np.random.default_rng(523)
    t = np.arange(1024)/16000
    train = [f'p1_{i}.wav' for i in range(4)]
    dev = ['p2_0.wav']
    for i, name in enumerate(train+dev):
        clean = (.08*np.sin(2*np.pi*(180+20*i)*t)).astype('float32')
        noisy = (clean+rng.normal(0, .01, len(t))).astype('float32')
        for part, wave in [('clean', clean), ('noisy', noisy)]:
            sf.write(out/'data'/f'{part}_trainset_28spk_wav'/name, wave, 16000, subtype='FLOAT')
    (out/'train.txt').write_text('\n'.join(train)+'\n')
    (out/'dev.txt').write_text('\n'.join(dev)+'\n')
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    for label, weight in [('baseline', '0'), ('clean_loss', '0.1')]:
        cmd = [sys.executable, str(ROOT/'scripts/train.py'), '--data-root', str(out/'data'),
               '--train-manifest', str(out/'train.txt'), '--dev-manifest', str(out/'dev.txt'),
               '--out', str(out/label), '--epochs', '1', '--batch-size', '4', '--workers', '0',
               '--crop-sec', '.064', '--dev-full-n', '1', '--device', 'cpu',
               '--clean-loss-weight', weight]
        result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
        (out/f'{label}.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
        if not (out/label/'best.safetensors').is_file():
            raise RuntimeError('No checkpoint saved')
    (out/'SMOKE_SUCCESS.json').write_text(json.dumps({
        'status': 'both training objectives completed one CPU optimizer step and exported weights',
        'data': 'synthetic tones; not speech, not a quality result, not a submission checkpoint',
    }, indent=2))
    print((out/'SMOKE_SUCCESS.json').read_text())


if __name__ == '__main__':
    main()
