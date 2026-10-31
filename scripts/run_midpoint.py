"""Run the chosen experiment and retain factual evidence, not report prose."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    out = a.run.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    stages = []

    def run(label, *args):
        started = time.time()
        command = [sys.executable, *map(str, args)]
        print('START', label, command, flush=True)
        with (out / f'{label}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        stages.append(dict(stage=label, exit_code=result.returncode,
                           seconds=time.time()-started, command=command))
        (out/'stages.json').write_text(json.dumps(stages, indent=2))
        print('END', label, 'exit', result.returncode, flush=True)
        return result.returncode == 0

    baseline = ROOT/'runs/baseline'
    for src, dst in [('train_files.txt', 'train.txt'), ('dev_files.txt', 'dev.txt')]:
        shutil.copy2(baseline/src, out/dst)
    hashes = {}
    for relative in ['scripts/train.py', 'scripts/measure_resources.py',
                     'scripts/evaluate_run.py', 'cli/submission/model.py']:
        source = ROOT/relative
        target = out/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'experiment.json').write_text(json.dumps({
        'hypothesis': 'Clean-to-clean mask loss may reduce clean-speech distortion.',
        'clean_loss_weight': .1, 'epochs': 30, 'seed': 0,
        'split': 'Team-chosen: p256 and p287 validation; course requirement unconfirmed',
        'comparison': 'Existing teammate baseline reports; original weights not independently rerun',
        'budget': 'Same noisy examples/epochs, extra clean forward pass; not compute matched',
        'source_sha256': hashes, 'server_submission': False,
    }, indent=2))
    if not run('resources', 'scripts/measure_resources.py', '--out', out/'resources.json',
               '--train-batch-sizes', '8', '--clean-loss-weight', '.1'):
        raise SystemExit('Resource measurement failed; training not started')
    resource = json.loads((out/'resources.json').read_text())
    memory = resource['train_peak_mem_GiB']['8']
    if 'L40S' not in resource['device'] or memory is None or memory > 32:
        raise SystemExit('Required L40S / 32 GiB memory check failed')
    if not run('training', 'scripts/train.py', '--data-root', os.environ['DATA_ROOT'],
               '--train-manifest', out/'train.txt', '--dev-manifest', out/'dev.txt',
               '--out', out/'model', '--epochs', '30', '--batch-size', '8',
               '--lr', '.001', '--crop-sec', '4', '--workers', '3', '--seed', '0',
               '--device', 'cuda', '--clean-loss-weight', '.1'):
        raise SystemExit('Training failed; inspect training.log')
    checkpoint = out/'model/best.safetensors'
    valid = run('output_checks', 'scripts/validate_checkpoint.py', '--checkpoint', checkpoint,
                '--device', 'cuda', '--out', out/'validation.json')
    # Preserve actual CLI evidence even if a synthetic edge case fails.
    evaluated = run('cli_evaluation', 'scripts/evaluate_run.py', '--checkpoint', checkpoint,
                    '--dev-dir', out/'model/dev', '--out', out/'evaluation', '--device', 'cuda')
    compared = evaluated and run('comparison', 'scripts/summarize_results.py',
                                '--run', baseline, '--comparison', out/'evaluation',
                                '--out', out/'comparison')
    if not (valid and evaluated and compared):
        raise SystemExit('NOT submission-ready: inspect stages.json and logs')
    bundle = out/'bundle/submission'
    bundle.mkdir(parents=True)
    shutil.copy2(checkpoint, bundle/'model.safetensors')
    shutil.copy2(out/'model/model_source.py', bundle/'model.py')
    (out/'LOCAL_PIPELINE_SUCCESS.json').write_text(json.dumps({
        'local_checks_passed': True, 'server_submitted': False,
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'bundle': str(bundle), 'report_written': False,
    }, indent=2))
    print('MIDPOINT_LOCAL_PIPELINE_SUCCESS (NOT a server receipt)', flush=True)


if __name__ == '__main__':
    main()
