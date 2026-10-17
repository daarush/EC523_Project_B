"""Data check: are noisy/clean pairs valid and time-aligned?"""
import random, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import correlate

root = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/projectnb/ec523kb/students/aarush/project_b/DeNoise/16khz")
clean_dir = root / "clean_trainset_28spk_wav"
noisy_dir = root / "noisy_trainset_28spk_wav"

clean_files = sorted(clean_dir.glob("*.wav"))
noisy_names = {p.name for p in noisy_dir.glob("*.wav")}
print(f"clean files: {len(clean_files)}, noisy files: {len(noisy_names)}")
print("extra noisy files with no clean match:",
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
print(f"Check 1 (format/length): {len(bad)} problems")
for b in bad[:20]: print("  ", b)

# Check 2: timing lag via cross-correlation on a random sample
lags = []
for c in random.Random(0).sample(clean_files, min(100, len(clean_files))):
    y, _ = sf.read(c); x, _ = sf.read(noisy_dir / c.name)
    n = min(len(x), len(y), 48000)
    cc = correlate(x[:n], y[:n], mode="full", method="fft")
    lags.append((int(np.argmax(cc)) - (n - 1), c.name))
off = [l for l in lags if abs(l[0]) > 2]
print(f"Check 2 (lag, {len(lags)} sampled): {len(off)} with |lag|>2 samples")
for l in off[:20]: print("  ", l)
print("max |lag|:", max(abs(l[0]) for l in lags))