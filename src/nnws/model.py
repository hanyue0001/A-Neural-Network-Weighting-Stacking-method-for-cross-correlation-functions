"""Neural network used by Neural Network Weighting Stacking (NNWS)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .sampling import resolve_sampling


class StackingNet(nn.Module):
    """Assign one weight to each CCF and return their normalized weighted stack.

    Parameters
    ----------
    lag
        Maximum absolute lag represented by each CCF.
    delta
        Legacy sampling interval in seconds per sample. If omitted, it is
        inferred from ``sampling_rate``.
    sampling_rate
        Sampling rate in Hz. Defaults to 1 Hz.
    hidden_dims
        Hidden-layer widths. The article architecture is used by default.
    eps
        Minimum denominator used by the normalized weighted mean.

    Notes
    -----
    The accepted input shapes are ``[M, N]`` and ``[1, M, N]``, where ``M`` is
    the number of CCFs and ``N = 2 * lag * sampling_rate + 1``. Batching
    multiple CCF matrices is intentionally rejected because each matrix has
    its own normalization denominator.
    """

    def __init__(
        self,
        lag: float = 500.0,
        delta: float | None = None,
        hidden_dims: Sequence[int] = (512, 256, 128, 64, 16),
        eps: float = 1e-8,
        sampling_rate: float = 1.0,
    ) -> None:
        super().__init__()
        delta, sampling_rate = resolve_sampling(delta, sampling_rate)
        if lag <= 0:
            raise ValueError("lag must be positive.")
        self.lag = float(lag)
        self.delta = delta
        self.sampling_rate = sampling_rate
        self.input_dim = int(round(2 * lag * sampling_rate)) + 1
        self.num_channels = self.input_dim  # legacy attribute
        self.eps = float(eps)

        widths = (self.input_dim, *tuple(hidden_dims), 1)
        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(widths[:-1], widths[1:])
        ):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.Sigmoid() if index == len(widths) - 2 else nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linear layers with the initialization used in this release."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )
                nn.init.zeros_(layer.bias)

    def _as_ccf_matrix(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[0] != 1:
                raise ValueError(
                    "NNWS expects one CCF matrix: [M, N] or [1, M, N]."
                )
            x = x.squeeze(0)
        elif x.ndim != 2:
            raise ValueError(
                f"NNWS expects [M, N] or [1, M, N], received {tuple(x.shape)}."
            )
        if x.shape[0] == 0:
            raise ValueError("At least one CCF is required.")
        input_dim = getattr(self, "input_dim", None)
        if input_dim is None:
            input_dim = next(
                layer.in_features for layer in self.net if isinstance(layer, nn.Linear)
            )
        if x.shape[1] != input_dim:
            raise ValueError(
                f"Expected N={input_dim} samples, received N={x.shape[1]}."
            )
        return x.to(dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(stack, weights)`` with shapes ``[1, N]`` and ``[M, 1]``."""
        ccf = self._as_ccf_matrix(x)
        weights = self.net(ccf)

        # Article definition: S = sum_i(w_i d_i) / sum_i(w_i).
        weight_sum = weights.sum(dim=0, keepdim=True).clamp_min(
            getattr(self, "eps", 1e-8)
        )
        stack = (weights.transpose(0, 1) @ ccf) / weight_sum
        return stack, weights


NeuralWeightStacker = StackingNet
