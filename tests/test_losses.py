import unittest

import torch

from nnws import SNRLoss, SymmetryLoss


class LossTest(unittest.TestCase):
    def test_symmetry_loss_is_zero_for_symmetric_signal(self):
        signal = torch.tensor([[1.0, 2.0, 4.0, 2.0, 1.0]])
        self.assertAlmostEqual(SymmetryLoss()(signal).item(), 0.0)

    def test_symmetry_loss_matches_rms_over_rms_definition(self):
        signal = torch.tensor([[1.0, 0.0, 2.0, 3.0, 4.0]])
        anti_rms = torch.sqrt(torch.mean((signal - signal.flip(-1)) ** 2, dim=-1))
        symmetric = signal[:, :3].flip(-1) + signal[:, 2:]
        symmetric_rms = torch.sqrt(torch.mean(symmetric**2, dim=-1))
        expected = (anti_rms / symmetric_rms).mean()
        torch.testing.assert_close(SymmetryLoss(eps=0)(signal), expected)

    def test_snr_loss_rejects_empty_windows(self):
        with self.assertRaises(ValueError):
            SNRLoss(tau=2, tmin=2, tmax=4)(torch.ones(1, 9))


if __name__ == "__main__":
    unittest.main()
