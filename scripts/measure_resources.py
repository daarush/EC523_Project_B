"""Resource measurements for the handout's required efficiency table.

project_b/scripts/. Run on the GPU you want to report (final numbers: L40S).
  python scripts/measure_resources.py --out runs/baseline/resources.json

Protocol (state this in the report):
  * inference: input [1,T] random noise (timing/memory do not depend on content),
    3 warm-up calls, 10 timed calls, torch.cuda.synchronize() around each call,
    report the median. RTF = processing seconds / audio seconds.
  * training memory: 3 real optimizer steps (Adam, same loss as train.py) on random
    4 s crops; value = torch.cuda.max_memory_allocated().
  * MACs: analytic. 1 MAC = 1 multiply-add. Counts only weight matrices of the LSTMs
    and linear layers (one MAC per weight per time step); ignores biases, activations,
    gates' elementwise ops, normalisation and STFT/iSTFT.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

SR = 16_000


def macs_per_second(model):
    fb = sum(p.numel() for p in model.fb_model.parameters() if p.ndim == 2)
    sb = sum(p.numel() for p in model.sb_model.parameters() if p.ndim == 2)
    n_freq = model.n_fft // 2 + 1
    per_frame = fb + n_freq * sb          # sub-band net runs once per frequency bin
    frames_per_s = SR / model.hop
    return {"fullband_weights": fb, "subband_weights": sb,
            "macs_per_frame": per_frame, "frames_per_second": frames_per_s,
            "GMACs_per_audio_second": per_frame * frames_per_s / 1e9,
            "share_of_macs_in_subband": n_freq * sb / per_frame}


def train_mem(build, dev, bs, crop, clean_weight=0.0):
    model = build(SR).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(3):
            x = torch.randn(bs, crop, device=dev) * 0.1
            y = torch.randn(bs, crop, device=dev) * 0.1
            pred, tgt = model.training_pair(x, y)
            loss = F.mse_loss(pred, tgt)
            if clean_weight:
                clean_pred, clean_target = model.training_pair(y, y)
                loss = loss + clean_weight * F.mse_loss(clean_pred, clean_target)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize()
        res = torch.cuda.max_memory_allocated() / 2 ** 30
    except torch.cuda.OutOfMemoryError:
        res = None
    del model, opt
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def infer(build, dev, secs, warmup=3, runs=10):
    model = build(SR).to(dev).eval()
    x = torch.randn(1, int(secs * SR), device=dev) * 0.1
    for _ in range(warmup):
        model.enhance(x, SR)
    if dev.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(runs):
        if dev.type == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        model.enhance(x, SR)
        if dev.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    return {"audio_s": secs, "median_latency_s": med, "rtf": med / secs,
            "peak_mem_GiB": (torch.cuda.max_memory_allocated() / 2 ** 30) if dev.type == "cuda" else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cli-dir", default=str(Path(__file__).resolve().parents[1] / "cli"))
    p.add_argument("--out", default="resources.json")
    p.add_argument("--train-batch-sizes", default="4,8,16")
    p.add_argument("--crop-sec", type=float, default=4.0)
    p.add_argument("--clean-loss-weight", type=float, default=0.0)
    a = p.parse_args()
    sys.path.insert(0, a.cli_dir)
    from submission.model import build_enhancer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "torch": torch.__version__, "clean_loss_weight": a.clean_loss_weight}
    m = build_enhancer(SR)
    res["params"] = sum(p.numel() for p in m.parameters())
    res["trainable_params"] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    res["macs"] = macs_per_second(m)
    del m
    res["inference"] = [infer(build_enhancer, dev, s) for s in (4, 10, 15)]
    if dev.type == "cuda":
        crop = int(a.crop_sec * SR)
        res["train_peak_mem_GiB"] = {int(b): train_mem(build_enhancer, dev, int(b), crop, a.clean_loss_weight)
                                     for b in a.train_batch_sizes.split(",")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
