# EC523 Project B — FullSubNet clean preservation

This branch adds a clean-input preservation objective to the team's FullSubNet
baseline. The network and inference interface are unchanged:

```text
loss = noisy compressed-mask MSE + 0.1 * clean-to-clean compressed-mask MSE
```

Only training pairs contribute gradients. The clean input uses the same training
crop as its noisy partner. The development set selects the checkpoint with the
lowest noisy mask loss, as in the baseline.

## Measured results

The October 30 run completed 30 epochs, all 15 synthetic output checks, and the
unchanged staff CLI on 736 noisy and 736 clean validation inputs.

| Metric | Recorded baseline | Clean-preservation experiment |
|---|---:|---:|
| Composite score | 65.2351 | 65.9056 |
| Noisy-input quality | 0.601462 | 0.598909 |
| Clean-input quality | 0.940725 | 0.999888 |
| Noisy-input PESQ | 2.428409 | 2.399315 |
| Noisy-input STOI | 0.859161 | 0.857342 |
| Noisy-input SI-SDR (dB) | 14.062263 | 14.237991 |
| L40S training memory, batch 8 (GiB) | 7.4276 | 12.4001 |
| Trainable parameters | 5,637,635 | 5,637,635 |

The composite improves by 0.6705 points because better clean preservation offsets
a small reduction in aggregate noisy-input quality. This is one seed on the same
team-chosen validation split, compared with the existing baseline reports; the
original baseline weights were not rerun. No server submission is claimed.
Epoch count and noisy examples are matched, but compute is not: the extra clean
pass increases training cost. Resource timings use synthetic inputs and fresh
instances of the unchanged architecture; see the measurement script's protocol.

Raw metrics, validation results, resource measurements, configuration, and training
log are in [runs/clean_preservation_20261030](runs/clean_preservation_20261030/).
Paired utterance intervals in `comparison/summary.json` do not measure uncertainty
across training seeds or speakers.

## Reproduce

Use Python 3.11 or 3.12 and install `requirements-training.txt`. The measured SCC
run used Python 3.12.4 and PyTorch 2.6.0+cu124 on an NVIDIA L40S. PESQ 0.0.4 and
pystoi 0.4.1 are pinned. Audio and model weights are not stored in Git.

```sh
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
PYTHONPATH=cli python -m unittest discover -s cli/tests -v
```

From the repository root, inside an allocated GPU session with the environment
activated:

```sh
export DATA_ROOT=/projectnb/ec523kb/projects/DeNoise/16khz
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -u scripts/run_midpoint.py --run runs/clean_preservation_reproduction
```

The orchestration script requires an L40S, measures memory for the extra objective,
trains, checks output constraints, runs the staff CLI, compares results, and creates
a model bundle only if all stages pass. Use a fresh output directory. `stages.json`
and separate logs retain stage failures. A local success file is not a server receipt.

To submit this workflow on SCC, create your environment first and use:

```sh
export VENV_PATH="$PWD/.venv-scc"
qsub -v VENV_PATH,RUN_DIR=runs/clean_preservation_reproduction train_clean_preservation.sh
```

The split is read explicitly from `runs/baseline/train_files.txt` and
`runs/baseline/dev_files.txt`: 10,836 training files from 26 speakers and 736
validation files from p256/p287. This reproduces the team's split; staff approval
of this split has not been independently established. `--clean-loss-weight 0`
in `scripts/train.py` disables the additional objective for a baseline control.

## Checkpoint and interface

The measured checkpoint SHA-256 is:

```text
7fb67c3fb9a081d5d2b76d84303ee8077802b9931977af828f66ecbd65a1ef10
```

It is stored on SCC at
`/projectnb/ec523kb/students/austinli/EC523_Project_B/runs/clean_preservation_20261030/bundle/submission/`.
Share the weights separately. Pair them with this branch's `cli/submission/model.py`.
The model source SHA-256 is recorded in the run configuration and validation output.
The public runner still requires finite mono 16 kHz output with identical length
and peak at most 1.0. No clipping or per-file normalization was added.

The implementation is adapted from
[official FullSubNet](https://github.com/Audio-WestlakeU/FullSubNet), revision
`e97448375cd1e883276ad583317b1828318910dc`; retain `LICENSE-FullSubNet` and cite the
paper and implementation. `cli/submission/model_prof.py` is the original untrained
interface demo. The handout and staff-owned CLI remain in `handout/` and `cli/`.
