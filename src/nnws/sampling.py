"""Sampling-parameter validation shared by NNWS components."""

from __future__ import annotations

import math


def resolve_sampling(
    delta: float | None = None, sampling_rate: float = 1.0
) -> tuple[float, float]:
    """Return a consistent ``(delta, sampling_rate)`` pair.

    ``sampling_rate`` is expressed in Hz and ``delta`` in seconds per sample.
    ``delta`` is retained for backward compatibility. When it is omitted,
    ``delta = 1 / sampling_rate``. A non-unit legacy ``delta`` supplied with
    the default sampling rate is interpreted as the older API and determines
    the effective sampling rate.
    """
    sampling_rate = float(sampling_rate)
    if not math.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be a positive finite value.")
    if delta is None:
        return 1.0 / sampling_rate, sampling_rate

    delta = float(delta)
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a positive finite value.")
    inferred_rate = 1.0 / delta
    if math.isclose(sampling_rate, 1.0) and not math.isclose(delta, 1.0):
        return delta, inferred_rate
    if not math.isclose(delta, 1.0 / sampling_rate, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "delta and sampling_rate conflict; require "
            "delta == 1 / sampling_rate."
        )
    return delta, sampling_rate
