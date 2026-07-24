"""NumPy helpers for applying learned NNWS weights."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def weighted_stack(
    data: ArrayLike, weights: ArrayLike, eps: float = 1e-8
) -> NDArray[np.floating]:
    """Compute ``sum_i(w_i d_i) / sum_i(w_i)`` for an ``[M, N]`` matrix."""
    matrix = np.asarray(data)
    weight_vector = np.asarray(weights).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("data must have shape [M, N].")
    if weight_vector.shape[0] != matrix.shape[0]:
        raise ValueError("One weight is required for every CCF.")
    denominator = float(weight_vector.sum())
    if denominator <= eps:
        raise ValueError("The weight sum is zero or too small.")
    return (weight_vector @ matrix) / denominator
