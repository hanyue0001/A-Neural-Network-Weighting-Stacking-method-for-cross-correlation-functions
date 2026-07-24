"""Open research implementation of Neural Network Weighting Stacking."""

from .losses import (
    Entropy_loss,
    SNRLoss,
    SymLoss,
    SymmetryLoss,
    WeightStdLoss,
    stdLoss,
)
from .metrics import (
    calculate_rms_ratio,
    get_symmetric_component,
    norm,
    normalize_max_abs,
    reported_snr,
    rms,
    rmsr_selective_stacking,
    rmsr_ss,
    selective_rms_ratio,
    training_rms_ratio,
)
from .model import NeuralWeightStacker, StackingNet
from .stacking import weighted_stack
from .training import (
    TrainingConfig,
    TrainingResult,
    apply_stacking_network,
    seed_everything,
    train,
    train_entire_dataset,
    train_stacking_network,
)

__all__ = [
    "Entropy_loss",
    "NeuralWeightStacker",
    "SNRLoss",
    "StackingNet",
    "SymLoss",
    "SymmetryLoss",
    "TrainingConfig",
    "TrainingResult",
    "WeightStdLoss",
    "apply_stacking_network",
    "calculate_rms_ratio",
    "get_symmetric_component",
    "norm",
    "normalize_max_abs",
    "reported_snr",
    "rms",
    "rmsr_selective_stacking",
    "rmsr_ss",
    "selective_rms_ratio",
    "seed_everything",
    "stdLoss",
    "train",
    "train_entire_dataset",
    "train_stacking_network",
    "training_rms_ratio",
    "weighted_stack",
]

__version__ = "0.1.0"
