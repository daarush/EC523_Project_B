"""Fail-fast evaluation through the unchanged staff CLI, into a fresh directory."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dev-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    a = p.parse_args()
    checkpoint, data, out = a.checkpoint.resolve(), a.dev_dir.resolve(), a.out.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not all((data / part).is_dir() for part in ('clean', 'noisy')):
        raise ValueError('dev-dir must contain clean/ and noisy/')
    if out.exists():
        raise ValueError('Choose a NEW output directory; old evidence will not be overwritten')
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'numpy', 'scipy', 'safetensors', 'pesq', 'pystoi', 'soundfile')}
    for name, version in [('pesq', '0.0.4'), ('pystoi', '0.4.1')]:
        if versions[name] != version:
            raise ValueError(f'{name} must be {version}')
    out.mkdir(parents=True)
    def run(*args):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT / 'cli',
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with (out / 'commands.log').open('a') as log:
            log.write(f'COMMAND: {args!r}\n{result.stdout}\n')
        print(result.stdout, end='', flush=True)
        result.check_returncode()
    for part, report in [('noisy', 'noisy_report.json'), ('clean', 'clean_preservation_report.json')]:
        run('enhance.py', '--checkpoint', checkpoint, '--input', data / part,
            '--output', out / f'pred_{part}', '--device', a.device)
        run('evaluate.py', '--clean', data / 'clean', '--enhanced', out / f'pred_{part}',
            '--report', out / report)
    run('evaluate.py', '--clean', data / 'clean', '--enhanced', data / 'noisy',
        '--report', out / 'noisy_input_report.json')
    run('score.py', '--noisy-report', out / 'noisy_report.json',
        '--clean-report', out / 'clean_preservation_report.json')
    # Written only after every command succeeds; incomplete runs have no SUCCESS.json.
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'SUCCESS.json').write_text(json.dumps({
        'status': 'local_evaluation_complete_NOT_server_submission',
        'checkpoint_sha256': digest(checkpoint),
        'source_sha256': digest(ROOT / 'cli/submission/model.py'),
        'versions': versions, 'device': a.device,
    }, indent=2))


if __name__ == '__main__':
    main()
