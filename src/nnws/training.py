"""Training and inference APIs for NNWS."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .losses import SNRLoss, SymmetryLoss, WeightStdLoss
from .model import StackingNet
from .sampling import resolve_sampling


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible training settings; defaults match the reported experiment."""

    epochs: int = 200
    learning_rate: float = 1e-3
    snr_weight: float = 0.8
    symmetry_weight: float = 0.2
    std_weight: float = 0.01
    seed: int = 0
    device: str = "auto"


@dataclass
class TrainingResult:
    model: StackingNet
    history: dict[str, list[float]]
    config: TrainingConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def train(
    data: np.ndarray,
    *,
    tau: float,
    tmin: float,
    tmax: float,
    delta: float | None = None,
    sampling_rate: float = 1.0,
    lag: float | None = None,
    config: TrainingConfig | None = None,
    save_path: str | Path | None = None,
    verbose: bool = False,
) -> TrainingResult:
    """Train NNWS on one ``[M, N]`` CCF matrix."""
    matrix = np.asarray(data)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] % 2 == 0:
        raise ValueError("data must be an [M, N] matrix with M >= 2 and odd N.")
    config = config or TrainingConfig()
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    delta, sampling_rate = resolve_sampling(delta, sampling_rate)
    seed_everything(config.seed)
    device = _device(config.device)
    inferred_lag = delta * (matrix.shape[1] - 1) / 2
    lag = inferred_lag if lag is None else lag
    model = StackingNet(
        lag=lag, delta=delta, sampling_rate=sampling_rate
    ).to(device)
    values = torch.as_tensor(matrix, dtype=torch.float32, device=device)

    snr_loss = SNRLoss(
        tau, tmin, tmax, delta, sampling_rate=sampling_rate
    ).to(device)
    symmetry_loss = SymmetryLoss().to(device)
    std_loss = WeightStdLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = {name: [] for name in ("total_loss", "snr_loss", "sym_loss", "std_loss")}

    model.train()
    for epoch in range(config.epochs):
        stack, weights = model(values)
        components = (
            snr_loss(stack),
            symmetry_loss(stack),
            std_loss(weights),
        )
        total = (
            config.snr_weight * components[0]
            + config.symmetry_weight * components[1]
            + config.std_weight * components[2]
        )
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        for key, value in zip(
            ("snr_loss", "sym_loss", "std_loss"), components
        ):
            history[key].append(float(value.detach().cpu()))
        history["total_loss"].append(float(total.detach().cpu()))
        if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
            print(
                f"Epoch {epoch + 1:>3}/{config.epochs}: "
                f"total={history['total_loss'][-1]:.6f}, "
                f"snr={history['snr_loss'][-1]:.6f}, "
                f"sym={history['sym_loss'][-1]:.6f}"
            )

    model.eval()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "lag": model.lag,
                "delta": model.delta,
                "sampling_rate": model.sampling_rate,
                "config": config.__dict__,
                "history": history,
            },
            path,
        )
    return TrainingResult(model=model, history=history, config=config)


@torch.no_grad()
def apply_stacking_network(
    model: StackingNet, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a trained model and return a 1-D stack and ``[M, 1]`` weights."""
    device = next(model.parameters()).device
    model.eval()
    stack, weights = model(torch.as_tensor(data, dtype=torch.float32, device=device))
    return stack.squeeze(0).cpu().numpy(), weights.cpu().numpy()


def train_stacking_network(
    data: np.ndarray,
    tau: float,
    tmin: float,
    tmax: float,
    num_epochs: int = 200,
    lag: float | None = None,
    delta: float | None = None,
    snr_weight: float = 0.8,
    sym_weight: float = 0.2,
    std_weight: float = 0.01,
    learning_rate: float = 1e-3,
    device: str = "auto",
    seed: int = 0,
    verbose: bool = False,
    sampling_rate: float = 1.0,
    **_: Any,
) -> tuple[StackingNet, dict[str, list[float]]]:
    """Compatibility wrapper around :func:`train`."""
    config = TrainingConfig(
        epochs=num_epochs,
        learning_rate=learning_rate,
        snr_weight=snr_weight,
        symmetry_weight=sym_weight,
        std_weight=std_weight,
        seed=seed,
        device=device,
    )
    result = train(
        data,
        tau=tau,
        tmin=tmin,
        tmax=tmax,
        delta=delta,
        sampling_rate=sampling_rate,
        lag=lag,
        config=config,
        verbose=verbose,
    )
    return result.model, result.history


def train_entire_dataset(
    num_epochs: int,
    l1: float,
    l2: float,
    lambda3: float,
    dataset: np.ndarray,
    tmin: float,
    tmax: float,
    tau: float,
    delta: float | None = None,
    lag: float = 500,
    *,
    seed: int = 0,
    device: str = "auto",
    sampling_rate: float = 1.0,
) -> tuple[StackingNet, float, float, float]:
    """Legacy signature retained for existing experiment scripts."""
    result = train(
        dataset,
        tau=tau,
        tmin=tmin,
        tmax=tmax,
        delta=delta,
        sampling_rate=sampling_rate,
        lag=lag,
        config=TrainingConfig(
            epochs=num_epochs,
            snr_weight=l1,
            symmetry_weight=l2,
            std_weight=lambda3,
            seed=seed,
            device=device,
        ),
    )
    history = result.history
    return (
        result.model,
        history["snr_loss"][-1],
        history["sym_loss"][-1],
        history["std_loss"][-1],
    )
