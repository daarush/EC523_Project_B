"""Data check: are noisy/clean pairs valid and time-aligned? Writes a txt report."""
import random, sys
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import correlate

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/projectnb/ec523kb/students/aarush/project_b/DeNoise/16khz")
report = Path(sys.argv[2] if len(sys.argv) > 2 else "alignment_report.txt")
N_LAG = None   # None = check every file; or set e.g. 100 for a random sample

out = open(report, "w")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); out.write(s + "\n")

clean_dir = root / "clean_trainset_28spk_wav"
noisy_dir = root / "noisy_trainset_28spk_wav"
clean_files = sorted(clean_dir.glob("*.wav"))
noisy_names = {p.name for p in noisy_dir.glob("*.wav")}
log(f"root: {root}")
log(f"clean files: {len(clean_files)}, noisy files: {len(noisy_names)}")
log("extra noisy files with no clean match:",
    len(noisy_names - {p.name for p in clean_files}))

# Check 1: existence, 16 kHz, mono, equal length
bad = []
for c in clean_files:
    if c.name not in noisy_names:
        bad.append((c.name, "missing noisy")); continue
    ci, ni = sf.info(c), sf.info(noisy_dir / c.name)
    if ci.samplerate != 16000 or ni.samplerate != 16000:
        bad.append((c.name, f"sr {ci.samplerate}/{ni.samplerate}"))
    elif ci.channels != 1 or ni.channels != 1:
        bad.append((c.name, f"channels {ci.channels}/{ni.channels}"))
    elif ci.frames != ni.frames:
        bad.append((c.name, f"length {ci.frames} vs {ni.frames}"))
log(f"Check 1 (format/length): {len(bad)} problems")
for b in bad[:20]: log("  ", b)

# Check 2 + 3: lag via cross-correlation, plus sanity of the samples
files = clean_files if N_LAG is None else random.Random(0).sample(
    clean_files, min(N_LAG, len(clean_files)))
lags, flags = [], []
for c in files:
    if c.name not in noisy_names: continue
    y, _ = sf.read(c); x, _ = sf.read(noisy_dir / c.name)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        flags.append((c.name, "non-finite"))
    if max(np.abs(x).max(), np.abs(y).max()) > 1.0:
        flags.append((c.name, "peak > 1.0"))
    if np.sqrt(np.mean(y ** 2)) < 1e-4:
        flags.append((c.name, "near-silent clean"))
    n = min(len(x), len(y), 48000)
    cc = correlate(x[:n], y[:n], mode="full", method="fft")
    lags.append((int(np.argmax(cc)) - (n - 1), c.name))

off = [l for l in lags if abs(l[0]) > 2]
log(f"Check 2 (lag, {len(lags)} files): {len(off)} with |lag|>2 samples")
for l in off[:20]: log("  ", l)
log("max |lag|:", max(abs(l[0]) for l in lags))
log("lag histogram:", dict(Counter(l for l, _ in lags)))
log(f"Check 3 (finite / peak<=1 / not silent): {len(flags)} flagged")
for f in flags[:20]: log("  ", f)
out.close()