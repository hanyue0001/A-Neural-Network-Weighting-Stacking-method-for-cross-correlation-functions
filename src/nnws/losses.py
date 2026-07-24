"""Self-supervised loss terms for NNWS."""

from __future__ import annotations

import torch
from torch import nn

from .sampling import resolve_sampling


def _as_signal_batch(signal: torch.Tensor) -> torch.Tensor:
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    if signal.ndim != 2:
        raise ValueError("Signal must have shape [N] or [B, N].")
    if signal.shape[-1] < 3 or signal.shape[-1] % 2 == 0:
        raise ValueError("A CCF must contain an odd number of at least 3 samples.")
    return signal


def symmetric_component(signal: torch.Tensor) -> torch.Tensor:
    """Return the zero-lag-aligned sum of the positive and negative branches."""
    signal = _as_signal_batch(signal)
    zero_lag = signal.shape[-1] // 2
    left = torch.flip(signal[:, : zero_lag + 1], dims=(-1,))
    right = signal[:, zero_lag:]
    return left + right


class SNRLoss(nn.Module):
    """Noise-RMS / signal-RMS using both off-signal branches as noise."""

    def __init__(
        self,
        tau: float,
        tmin: float,
        tmax: float,
        delta: float | None = None,
        eps: float = 1e-8,
        sampling_rate: float = 1.0,
    ) -> None:
        super().__init__()
        delta, sampling_rate = resolve_sampling(delta, sampling_rate)
        if tau < 0 or tmin < 0 or tmax <= tmin:
            raise ValueError("Require tau >= 0 and 0 <= tmin < tmax.")
        self.delta = delta
        self.sampling_rate = sampling_rate
        self.tau_idx = int(tau / delta)
        self.tmin_idx = int(tmin / delta)
        self.tmax_idx = int(tmax / delta)
        self.eps = float(eps)

    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x.square(), dim=-1))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        sym = symmetric_component(signal)
        start = self.tmin_idx - 2 * self.tau_idx
        stop = self.tmax_idx + 2 * self.tau_idx
        if start <= 0 or stop >= sym.shape[-1] or start >= stop:
            raise ValueError(
                "The signal window must be non-empty and leave noise samples "
                "on both sides; check tau, tmin, tmax, sampling_rate, and "
                "CCF length."
            )
        signal_window = sym[:, start:stop]
        noise_window = torch.cat((sym[:, :start], sym[:, stop:]), dim=-1)
        return (self._rms(noise_window) / (self._rms(signal_window) + self.eps)).mean()


class SymmetryLoss(nn.Module):
    """Anti-symmetric RMS divided by symmetric-component RMS.

    This is the RMS/RMS interpretation of :math:`L_\\mathrm{SYM}`. The
    implementation, variable names, and documentation deliberately use the
    same definition; no squared-energy quantity is implied.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal = _as_signal_batch(signal)
        anti_symmetric_rms = torch.sqrt(
            torch.mean((signal - torch.flip(signal, dims=(-1,))).square(), dim=-1)
        )
        symmetric_rms = torch.sqrt(
            torch.mean(symmetric_component(signal).square(), dim=-1)
        )
        return (anti_symmetric_rms / (symmetric_rms + self.eps)).mean()


class WeightStdLoss(nn.Module):
    """Negative weight standard deviation, encouraging differentiated weights."""

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        if weights.numel() < 2:
            raise ValueError("WeightStdLoss requires at least two weights.")
        return -torch.std(weights)


# Backward-compatible names used by the research scripts.
SymLoss = SymmetryLoss
stdLoss = WeightStdLoss
Entropy_loss = WeightStdLoss
