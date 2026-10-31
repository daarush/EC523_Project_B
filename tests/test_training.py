import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'cli'))
sys.path.insert(0, str(ROOT / 'scripts'))
from train import load_split, training_loss
from submission.model import FullSubNet, build_compressed_cirm, decompress_cirm

torch.set_num_threads(2)


class TrainingChecks(unittest.TestCase):
    def test_mask_reconstruction(self):
        # Known nonzero noisy complex values and a known ratio exercise both signs.
        noisy = torch.tensor([[[1+2j, -2+1j]]])
        ratio = torch.tensor([[[.7+.2j, 1.1-.3j]]])
        clean = noisy*ratio
        mask = decompress_cirm(build_compressed_cirm(noisy, clean))
        reconstructed = noisy*torch.complex(mask[:, 0], mask[:, 1])
        torch.testing.assert_close(reconstructed, clean, atol=3e-6, rtol=3e-6)

    def test_clean_objective_has_finite_gradients(self):
        torch.manual_seed(523)
        model = FullSubNet(fb_hidden=8, sb_hidden=8, weight_init=False).train()
        clean = torch.randn(4, 1024)*.05
        noisy = clean + torch.randn_like(clean)*.02
        base = training_loss(model, noisy, clean, 0)
        pred, target = model.training_pair(clean, clean)
        expected = base + .1*torch.nn.functional.mse_loss(pred, target)
        combined = training_loss(model, noisy, clean, .1)
        torch.testing.assert_close(combined, expected)
        combined.backward()
        for network in (model.fb_model, model.sb_model):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in network.parameters()), 0)

    def test_identity_mask_preserves_tail_and_length(self):
        # Replace only mask prediction with the exact identity to isolate STFT/iSTFT padding.
        model = FullSubNet(fb_hidden=8, sb_hidden=8, weight_init=False).eval()
        def identity(mag, drop):
            out = torch.zeros(mag.shape[0], 2, mag.shape[2], mag.shape[3])
            out[:, 0] = 10*torch.tanh(torch.tensor(.05))
            return out
        model._net = identity
        for n in (1, 100, 255, 256, 257, 511, 512, 513, 16001):
            x = torch.randn(1, n)*.1
            y = model.enhance(x, 16000)
            torch.testing.assert_close(y, x, atol=2e-6, rtol=2e-5)

    def test_split_rejects_speaker_overlap_and_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); clean = root/'clean'; noisy = root/'noisy'
            clean.mkdir(); noisy.mkdir()
            for folder in (clean, noisy):
                for name in ('p1_a.wav', 'p1_b.wav', 'p2_a.wav'):
                    sf.write(folder/name, np.ones(512)*.01, 16000)
            tr, dv = root/'train.txt', root/'dev.txt'
            tr.write_text('p1_a.wav\n'); dv.write_text('p1_b.wav\n')
            with self.assertRaisesRegex(ValueError, 'speaker overlap'):
                load_split(tr, dv, clean, noisy)
            dv.write_text('p2_a.wav\n')
            self.assertEqual(load_split(tr, dv, clean, noisy), (['p1_a.wav'], ['p2_a.wav']))
            sf.write(noisy/'p2_a.wav', np.ones(513)*.01, 16000)
            with self.assertRaisesRegex(ValueError, 'unequal lengths'):
                load_split(tr, dv, clean, noisy)


if __name__ == '__main__':
    unittest.main()
