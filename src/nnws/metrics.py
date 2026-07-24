"""Signal-processing metrics and the RMSR selective-stacking baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .sampling import resolve_sampling


def rms(data: ArrayLike) -> float:
    values = np.asarray(data)
    if values.size == 0:
        raise ValueError("RMS is undefined for an empty window.")
    return float(np.sqrt(np.mean(values**2)))


def normalize_max_abs(
    data: ArrayLike, axis: int = -1, eps: float = 1e-12
) -> NDArray[np.floating]:
    values = np.asarray(data)
    scale = np.max(np.abs(values), axis=axis, keepdims=True)
    return values / np.maximum(scale, eps)


def get_symmetric_component(data: ArrayLike) -> NDArray[np.floating]:
    values = np.asarray(data)
    if values.ndim != 1 or values.size < 3 or values.size % 2 == 0:
        raise ValueError("Expected one odd-length CCF.")
    zero_lag = values.size // 2
    return values[: zero_lag + 1][::-1] + values[zero_lag:]


def _window_indices(
    length: int, tau: float, tmin: float, tmax: float, delta: float
) -> tuple[int, int]:
    if delta <= 0 or tau < 0 or tmin < 0 or tmax <= tmin:
        raise ValueError("Require delta > 0, tau >= 0, and 0 <= tmin < tmax.")
    start = int(tmin / delta) - 2 * int(tau / delta)
    stop = int(tmax / delta) + 2 * int(tau / delta)
    if start <= 0 or stop >= length or start >= stop:
        raise ValueError("Signal/noise windows are empty or outside the CCF.")
    return start, stop


def training_rms_ratio(
    data: ArrayLike,
    tau: float,
    tmin: float,
    tmax: float,
    delta: float | None = None,
    sampling_rate: float = 1.0,
) -> float:
    """Signal/noise RMS ratio matching ``SNRLoss`` (both noise branches)."""
    delta, _ = resolve_sampling(delta, sampling_rate)
    symmetric = get_symmetric_component(data)
    start, stop = _window_indices(len(symmetric), tau, tmin, tmax, delta)
    noise = np.concatenate((symmetric[:start], symmetric[stop:]))
    return rms(symmetric[start:stop]) / rms(noise)


def reported_snr(
    data: ArrayLike,
    tau: float,
    tmin: float,
    tmax: float,
    delta: float | None = None,
    sampling_rate: float = 1.0,
) -> float:
    """Reported SNR as signal-window RMS / remote-tail noise RMS."""
    delta, _ = resolve_sampling(delta, sampling_rate)
    symmetric = get_symmetric_component(data)
    start, stop = _window_indices(len(symmetric), tau, tmin, tmax, delta)
    tau_idx = int(tau / delta)
    tmax_idx = int(tmax / delta)
    tail = symmetric[tmax_idx + 4 * tau_idx : tmax_idx + 8 * tau_idx]
    if tail.size == 0:
        raise ValueError("The reported tail-noise window is empty.")
    return rms(symmetric[start:stop]) / rms(tail)


def selective_rms_ratio(
    data: ArrayLike,
    tau: float,
    tmin: float,
    tmax: float,
    delta: float | None = None,
    sampling_rate: float = 1.0,
) -> float:
    """RMSR_SS criterion using pre-signal noise plus the remote tail window."""
    delta, _ = resolve_sampling(delta, sampling_rate)
    symmetric = get_symmetric_component(data)
    start, stop = _window_indices(len(symmetric), tau, tmin, tmax, delta)
    tau_idx = int(tau / delta)
    tmax_idx = int(tmax / delta)
    remote_tail = symmetric[
        tmax_idx + 4 * tau_idx : tmax_idx + 8 * tau_idx
    ]
    if remote_tail.size == 0:
        raise ValueError("The RMSR_SS tail-noise window is empty.")
    noise = np.concatenate((symmetric[:start], remote_tail))
    return rms(symmetric[start:stop]) / rms(noise)


def rmsr_selective_stacking(
    data: ArrayLike,
    tau: float,
    tmin: float,
    tmax: float,
    delta: float | None = None,
    threshold: float = 1.0,
    sampling_rate: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]:
    """Return linear stack, RMSR selective stack, and selected-CCF mask.

    The threshold is explicitly parameterized as ``G = 1 + threshold / M`` so
    figure scripts can state whether they use 1 or 5.
    """
    matrix = np.asarray(data)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("data must be a non-empty [M, N] CCF matrix.")
    delta, _ = resolve_sampling(delta, sampling_rate)
    linear = matrix.sum(axis=0)
    reference = selective_rms_ratio(linear, tau, tmin, tmax, delta)
    limit = 1.0 + threshold / matrix.shape[0]
    selected = np.zeros(matrix.shape[0], dtype=bool)
    for index, ccf in enumerate(matrix):
        candidate = selective_rms_ratio(
            linear - ccf, tau, tmin, tmax, delta
        )
        selected[index] = candidate / reference <= limit if reference > 0 else False
    selective = matrix[selected].sum(axis=0) if selected.any() else linear.copy()
    return linear, selective, selected


# Compatibility aliases for the RMSR_SS baseline and normalization helper.
norm = normalize_max_abs
calculate_rms_ratio = selective_rms_ratio
rmsr_ss = rmsr_selective_stacking
