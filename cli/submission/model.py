"""FullSubNet in the course interface (self-contained, single file).

Adapted from the official implementation (Audio-WestlakeU/FullSubNet,
recipes/dns_interspeech_2020/fullsubnet/model.py + audio_zen base_model.py,
sequence_model.py, feature.py). Cite it in the report and check the repo LICENSE.
Only the code path used by the paper's config is kept
(offline_laplace_norm, LSTM, fb_num_neighbors=0, sb_num_neighbors=15, look_ahead=2).

Course interface: build_enhancer(16000) -> module with enhance([1,T], 16000) -> [1,T].

The cIRM build/compress/decompress equations and EPS were checked against the
locally cloned official audio_zen/acoustics/mask.py and constant.py on 2026-10-30.
This implementation puts the real/imaginary dimension second rather than last.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

EPS = torch.finfo(torch.float32).eps  # Same float32 epsilon as official NumPy constant.


# ---------------------------------------------------------------- helpers (official)
def offline_laplace_norm(x: torch.Tensor) -> torch.Tensor:
    """Divide by ONE scalar mean per sample (mean over every dim except batch)."""
    mu = torch.mean(x, dim=list(range(1, x.dim())), keepdim=True)
    return x / (mu + 1e-5)


def freq_unfold(x: torch.Tensor, num_neighbors: int) -> torch.Tensor:
    """[B,1,F,T] -> [B,F,2n+1,T]: each frequency with n neighbours per side.
    Official code pads with REFLECT (the paper text says circular). n<=0: no-op reshape."""
    if num_neighbors <= 0:
        return x.permute(0, 2, 1, 3)                                   # [B,F,1,T]
    x = F.pad(x, [0, 0, num_neighbors, num_neighbors], mode="reflect")  # [B,1,F+2n,T]
    x = x.squeeze(1).unfold(1, 2 * num_neighbors + 1, 1)               # [B,F,T,2n+1]
    return x.permute(0, 1, 3, 2)                                       # [B,F,2n+1,T]


def drop_band(x: torch.Tensor, num_groups: int = 2) -> torch.Tensor:
    """[B,C,F,T] -> [B,C,F//g,T]. Deterministic: samples in group k keep bins k, k+g, ...
    Output batch is REORDERED (group 0 samples first). Needs B > num_groups."""
    batch_size, _, num_freqs, _ = x.shape
    assert batch_size > num_groups, f"batch {batch_size} must be > num_groups {num_groups}"
    if num_groups <= 1:
        return x
    if num_freqs % num_groups != 0:                      # 257 -> 256: top bin is dropped
        x = x[..., : num_freqs - (num_freqs % num_groups), :]
        num_freqs = x.shape[2]
    out = []
    for g in range(num_groups):
        s_idx = torch.arange(g, batch_size, num_groups, device=x.device)
        f_idx = torch.arange(g, num_freqs, num_groups, device=x.device)
        out.append(torch.index_select(torch.index_select(x, 0, s_idx), 2, f_idx))
    return torch.cat(out, dim=0)


class SequenceModel(nn.Module):
    """LSTM (unidirectional) + Linear (+ optional ReLU). In/out: [B, feat, T]."""

    def __init__(self, input_size, output_size, hidden_size, num_layers, relu_out):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu_out = relu_out

    def forward(self, x):
        o, _ = self.rnn(x.permute(0, 2, 1))   # [B,T,feat]
        o = self.fc(o)
        if self.relu_out:
            o = F.relu(o)
        return o.permute(0, 2, 1)             # [B,feat,T]


def _official_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for p in m.parameters():
            if len(p.shape) >= 2:
                init.orthogonal_(p.data)
            else:
                init.normal_(p.data)


# ---------------------------------------------------------------- cIRM
K_MASK, C_MASK, LIMIT = 10.0, 0.1, 9.9


def build_compressed_cirm(noisy_spec, clean_spec):
    """complex [B,F,T] x2 -> compressed cIRM [B,2,F,T] (real, imag)."""
    xr, xi, yr, yi = noisy_spec.real, noisy_spec.imag, clean_spec.real, clean_spec.imag
    den = xr ** 2 + xi ** 2 + EPS
    m = torch.stack(((xr * yr + xi * yi) / den, (xr * yi - xi * yr) / den), dim=1)
    m = m.clamp(min=-100)
    return K_MASK * (1 - torch.exp(-C_MASK * m)) / (1 + torch.exp(-C_MASK * m))


def decompress_cirm(y):
    y = y.clamp(-LIMIT, LIMIT)
    return -K_MASK * torch.log((K_MASK - y) / (K_MASK + y))


# ---------------------------------------------------------------- model
class FullSubNet(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        hop: int = 256,              # paper/official; course handout baseline uses 128
        look_ahead: int = 2,
        sb_num_neighbors: int = 15,
        fb_num_neighbors: int = 0,
        fb_hidden: int = 512,
        sb_hidden: int = 384,
        num_groups_in_drop_band: int = 2,
        weight_init: bool = True,
    ):
        super().__init__()
        if sample_rate != 16_000:
            raise ValueError("the course interface is fixed at 16 kHz")
        self.n_fft, self.hop = n_fft, hop
        n_freq = n_fft // 2 + 1
        self.look_ahead = look_ahead
        self.sb_nb, self.fb_nb = sb_num_neighbors, fb_num_neighbors
        self.groups = num_groups_in_drop_band
        self.register_buffer("window", torch.hann_window(n_fft), persistent=True)

        self.fb_model = SequenceModel(n_freq, n_freq, fb_hidden, 2, relu_out=True)
        self.sb_model = SequenceModel(
            (2 * sb_num_neighbors + 1) + (2 * fb_num_neighbors + 1), 2, sb_hidden, 2, relu_out=False
        )
        if weight_init:
            self.apply(_official_init)

    # --- STFT
    def stft(self, wave):
        return torch.stft(wave, self.n_fft, self.hop, self.n_fft, window=self.window,
                          return_complex=True)

    def istft(self, spec, length):
        return torch.istft(spec, self.n_fft, self.hop, self.n_fft, window=self.window,
                           length=length)

    # --- the network: [B,1,F,T] magnitude -> compressed cIRM [B,2,F',T]
    def _net(self, noisy_mag, drop: bool):
        noisy_mag = F.pad(noisy_mag, [0, self.look_ahead])            # look-ahead pad (end)
        B, _, Fq, T = noisy_mag.shape

        fb_in = offline_laplace_norm(noisy_mag).reshape(B, Fq, T)
        fb_out = self.fb_model(fb_in).reshape(B, 1, Fq, T)

        fb_unf = freq_unfold(fb_out, self.fb_nb)                       # [B,F,2fb+1,T]
        nz_unf = freq_unfold(noisy_mag, self.sb_nb)                    # [B,F,2sb+1,T]
        sb_in = offline_laplace_norm(torch.cat([nz_unf, fb_unf], dim=2))   # [B,F,32,T]

        if drop:   # training-speed trick; output then has F//groups bins, batch reordered
            sb_in = drop_band(sb_in.permute(0, 2, 1, 3), self.groups).permute(0, 2, 1, 3)

        B2, Fp, D, T = sb_in.shape
        sb_out = self.sb_model(sb_in.reshape(B2 * Fp, D, T))           # [B*F',2,T]
        out = sb_out.reshape(B2, Fp, 2, T).permute(0, 2, 1, 3)         # [B,2,F',T]
        return out[..., self.look_ahead:]

    # --- waveform -> waveform (full bandwidth, never drops bands)
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        length = wave.shape[-1]
        padded_len = -(-(length + self.n_fft // 2) // self.hop) * self.hop
        wave = F.pad(wave, (0, padded_len - length))
        spec = self.stft(wave)
        comp = self._net(spec.abs().unsqueeze(1), drop=False)
        m = decompress_cirm(comp)
        out = self.istft(spec * torch.complex(m[:, 0], m[:, 1]), padded_len)
        return out[..., :length]

    # --- training: loss = F.mse_loss(*model.training_pair(noisy, clean))
    def training_pair(self, noisy: torch.Tensor, clean: torch.Tensor):
        ns, cs = self.stft(noisy), self.stft(clean)
        drop = self.training and self.groups > 1 and noisy.shape[0] > self.groups
        pred = self._net(ns.abs().unsqueeze(1), drop=drop)
        target = build_compressed_cirm(ns, cs)
        if drop:
            target = drop_band(target, self.groups)   # same reorder/selection as the input
        return pred, target

    @torch.no_grad()
    def enhance(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != 16_000 or waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("expected one [1,T] waveform at 16 kHz")
        return self.forward(waveform)


def build_enhancer(sample_rate: int) -> nn.Module:
    return FullSubNet(sample_rate=sample_rate)
