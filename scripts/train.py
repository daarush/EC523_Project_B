"""Train the FullSubNet baseline on the 28-speaker paired set.

project_b/scripts/ (it imports cli/submission/model.py).

  # 1) overfit sanity check (should drive train loss way down on 64 files)
  python -u scripts/train.py --out runs/overfit --limit-files 64 --epochs 40 --batch-size 8
  # 2) real run
  python -u scripts/train.py --out runs/baseline --epochs 30 --batch-size 8

Everything that must stay IDENTICAL between baseline and modified runs is a CLI flag
or fixed constant here: split seed, crop, optimizer, lr, batch size, epochs, selection rule.
Selection rule (declared): checkpoint with the lowest dev loss (MSE on compressed cIRM).
"""
import argparse, json, random, shutil, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

SR = 16_000


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/projectnb/ec523kb/students/aarush/project_b/DeNoise/16khz")
    p.add_argument("--cli-dir", default=str(Path(__file__).resolve().parents[1] / "cli"))
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)   # must be > num_groups_in_drop_band (2)
    p.add_argument("--lr", type=float, default=1e-3)       # paper: Adam, 1e-3
    p.add_argument("--clip", type=float, default=10.0)     # my default; not in the files you pasted
    p.add_argument("--crop-sec", type=float, default=4.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)          # training seed
    p.add_argument("--split-seed", type=int, default=0)    # keep fixed across ALL runs
    p.add_argument("--n-dev-speakers", type=int, default=2)
    p.add_argument("--dev-speakers", default="")           # e.g. p226,p227 (overrides random pick)
    p.add_argument("--limit-files", type=int, default=0)   # >0: only first N train files (overfit test)
    p.add_argument("--dev-full-n", type=int, default=40)   # full utterances for SI-SDR/peak logging
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


class PairCrops(Dataset):
    """Same start index for noisy and clean. Shorter files are zero-padded together."""

    def __init__(self, clean_dir, noisy_dir, names, lengths, crop, fixed):
        self.cd, self.nd, self.names, self.lengths = clean_dir, noisy_dir, names, lengths
        self.crop, self.fixed = crop, fixed

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        L, n = self.lengths[i], self.names[i]
        if L > self.crop:
            hi = L - self.crop + 1
            start = (i * 7919) % hi if self.fixed else int(torch.randint(0, hi, (1,)))
        else:
            start = 0
        stop = min(start + self.crop, L)
        y, _ = sf.read(self.cd / n, start=start, stop=stop, dtype="float32")
        x, _ = sf.read(self.nd / n, start=start, stop=stop, dtype="float32")
        pad = self.crop - len(y)
        if pad > 0:
            y, x = np.pad(y, (0, pad)), np.pad(x, (0, pad))
        return torch.from_numpy(x), torch.from_numpy(y)


def si_sdr(est, ref, eps=1e-8):
    est = est.astype(np.float64) - est.mean()
    ref = ref.astype(np.float64) - ref.mean()
    a = (est @ ref) / (ref @ ref + 1e-12)
    t = a * ref
    r = est - t
    return 10 * np.log10((t @ t + eps) / (r @ r + eps))


def save_safetensors(model, path):
    # LSTM params can be views of one cuDNN buffer; safetensors refuses shared storage.
    sd = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(path))


def main():
    a = get_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, a.cli_dir)
    from submission.model import build_enhancer

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(a.data_root)
    cd, nd = root / "clean_trainset_28spk_wav", root / "noisy_trainset_28spk_wav"
    names = sorted(p.name for p in cd.glob("*.wav"))
    assert all((nd / n).exists() for n in names), "missing noisy file"

    # ---- speaker-grouped split (depends only on --split-seed / --dev-speakers)
    spk = lambda n: n.split("_")[0]
    speakers = sorted({spk(n) for n in names})
    dev_spk = (a.dev_speakers.split(",") if a.dev_speakers
               else sorted(random.Random(a.split_seed).sample(speakers, a.n_dev_speakers)))
    tr_names = [n for n in names if spk(n) not in dev_spk]
    dv_names = [n for n in names if spk(n) in dev_spk]
    (out / "dev_speakers.txt").write_text("\n".join(dev_spk) + "\n")
    (out / "train_files.txt").write_text("\n".join(tr_names) + "\n")
    (out / "dev_files.txt").write_text("\n".join(dv_names) + "\n")
    if a.limit_files:
        tr_names = tr_names[: a.limit_files]
    print(f"speakers {len(speakers)}, dev {dev_spk}, train files {len(tr_names)}, dev files {len(dv_names)}", flush=True)

    # ---- file lengths (cached: sf.info over network FS is slow)
    cache = Path("lengths_28spk.json")
    lens = json.loads(cache.read_text()) if cache.exists() else {}
    if any(n not in lens for n in names):
        print("reading file lengths (first run only)...", flush=True)
        lens = {n: sf.info(cd / n).frames for n in names}
        cache.write_text(json.dumps(lens))
    crop = int(a.crop_sec * SR)

    tr_ds = PairCrops(cd, nd, tr_names, [lens[n] for n in tr_names], crop, fixed=False)
    dv_ds = PairCrops(cd, nd, dv_names, [lens[n] for n in dv_names], crop, fixed=True)
    g = torch.Generator(); g.manual_seed(a.seed)
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers,
                       generator=g, pin_memory=True, persistent_workers=a.workers > 0)
    dv_dl = DataLoader(dv_ds, 16, shuffle=False, num_workers=a.workers)

    # ---- export dev set in the CLI's layout, so the OFFICIAL enhance/evaluate/score can be run on it
    for sub, src in (("noisy", nd), ("clean", cd)):
        d = out / "dev" / sub; d.mkdir(parents=True, exist_ok=True)
        for n in dv_names:
            if not (d / n).exists():
                shutil.copy(src / n, d / n)

    model = build_enhancer(SR).to(dev)
    assert a.batch_size > model.groups, "batch size must exceed num_groups_in_drop_band"
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n_params = sum(p.numel() for p in model.parameters())
    (out / "config.json").write_text(json.dumps({**vars(a), "dev_speakers": dev_spk,
                                                 "n_params": n_params}, indent=2))
    print(f"params {n_params/1e6:.3f}M, device {dev}", flush=True)

    start_epoch, best = 0, float("inf")
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start_epoch}", flush=True)

    dev_full = dv_names[:: max(1, len(dv_names) // max(1, a.dev_full_n))][: a.dev_full_n]
    log = open(out / "log.txt", "a")

    for epoch in range(start_epoch, a.epochs):
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        tot, nb = 0.0, 0
        for step, (x, y) in enumerate(tr_dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            pred, tgt = model.training_pair(x, y)
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            tot += loss.item(); nb += 1
            if step % 100 == 0:
                print(f"  ep {epoch} step {step}/{len(tr_dl)} loss {loss.item():.4f}", flush=True)
        tr_loss = tot / max(nb, 1)

        # ---- dev: fixed crops loss + full-utterance SI-SDR / peak
        model.eval()
        with torch.no_grad():
            s, n = 0.0, 0
            for x, y in dv_dl:
                x, y = x.to(dev), y.to(dev)
                pred, tgt = model.training_pair(x, y)
                s += F.mse_loss(pred, tgt, reduction="sum").item(); n += pred.numel()
            dv_loss = s / n
            sd_in, sd_out, peak = [], [], 0.0
            for nme in dev_full:
                yy, _ = sf.read(cd / nme, dtype="float32"); xx, _ = sf.read(nd / nme, dtype="float32")
                en = model.enhance(torch.from_numpy(xx)[None].to(dev), SR)[0].cpu().numpy()
                sd_in.append(si_sdr(xx, yy)); sd_out.append(si_sdr(en, yy))
                peak = max(peak, float(np.abs(en).max()))
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        line = (f"epoch {epoch} train {tr_loss:.4f} dev {dv_loss:.4f} "
                f"dev_SISDR noisy {np.mean(sd_in):.2f} -> enh {np.mean(sd_out):.2f} dB "
                f"peak {peak:.3f} mem {mem:.1f}GiB time {time.time()-t0:.0f}s")
        print(line, flush=True); log.write(line + "\n"); log.flush()

        if dv_loss < best:
            best = dv_loss
            save_safetensors(model, out / "best.safetensors")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best": best}, out / "last.pt")
    print("done. best dev loss", best)


if __name__ == "__main__":
    main()