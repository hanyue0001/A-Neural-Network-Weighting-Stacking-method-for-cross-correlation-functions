"""
NeuralStacking - A PyTorch-based neural network for seismic signal stacking
================================================================================

This module implements a deep learning approach for improving signal-to-noise ratio
in seismic cross-correlation functions through intelligent stacking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Union, List
import warnings

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"


# ==============================================================================
# Signal Processing Functions
# ==============================================================================

def rms(data: np.ndarray) -> float:
    """Calculate the root mean square of an array."""
    return np.sqrt(np.mean(data**2))


def stack_all(data: np.ndarray) -> np.ndarray:
    """Stack all input signals along the first axis."""
    return np.sum(data, axis=0)


def get_symmetric_component(data: np.ndarray) -> np.ndarray:
    """Extract symmetric component from a signal."""
    length = len(data)
    zero_lag = length // 2
    
    left = data[:zero_lag+1]
    right = data[zero_lag:]
    left = left[::-1]
    
    return np.sum(np.vstack((left, right)), axis=0)


def calculate_rms_ratio(data: np.ndarray, tau: float, tmin: float, 
                        tmax: float, delta: float = 1.0) -> float:
    """
    Calculate RMS ratio between signal window and noise window.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal
    tau : float
        Time lag parameter
    tmin : float
        Start time of signal window
    tmax : float
        End time of signal window
    delta : float, default=1.0
        Sampling interval
        
    Returns
    -------
    float
        RMS ratio (signal RMS / noise RMS)
    """
    tau_idx = int(tau/delta)
    tmin_idx = int(tmin/delta)
    tmax_idx = int(tmax/delta)
    
    syme = get_symmetric_component(data)
    
    # Define signal and noise windows
    signal_win = syme[tmin_idx-2*tau_idx:tmax_idx+2*tau_idx]
    noise_win1 = syme[:tmin_idx-2*tau_idx]
    noise_win2 = syme[tmax_idx+4*tau_idx:tmax_idx+8*tau_idx]
    noise_combined = np.hstack((noise_win1, noise_win2))
    
    # Calculate RMS ratio
    if len(signal_win) > 0 and len(noise_combined) > 0:
        return rms(signal_win) / rms(noise_combined)
    else:
        return 0.0


def rmsr_selective_stacking(data: np.ndarray, tau: float, tmin: float, 
                           tmax: float, delta: float = 1.0, threshold:float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform selective stacking based on RMS ratio criterion.
    
    Parameters
    ----------
    data : np.ndarray
        Input cross-correlation functions, shape (n_segments, signal_length)
    tau : float
        Time lag parameter
    tmin : float
        Start time of signal window
    tmax : float
        End time of signal window
    delta : float, default=1.0
        Sampling interval
        
    Returns
    -------
    linear_stack : np.ndarray
        Linear stack of all signals
    selective_stack : np.ndarray
        Stack of selected signals
    selection_mask : np.ndarray
        Boolean mask indicating which signals were selected
    """
    n_segments = len(data)
    
    # Step 1: Linear stacking
    linear_stack = stack_all(data)
    rn = calculate_rms_ratio(linear_stack, tau, tmin, tmax, delta)
    
    # Step 2: Selective stacking
    selected_data = []
    selected_mask = np.zeros(n_segments, dtype=bool)
    G = 1 + threshold / n_segments  # Selection threshold
    
    for i in range(n_segments):
        temp_stack = linear_stack - data[i]
        rk = calculate_rms_ratio(temp_stack, tau, tmin, tmax, delta)
        
        Q = rk / rn if rn > 0 else 0
        
        if Q <= G:
            selected_mask[i] = True
            selected_data.append(data[i])
    
    # Stack selected signals
    if len(selected_data) > 0:
        selective_stack = stack_all(selected_data)
    else:
        selective_stack = linear_stack.copy()
    
    return linear_stack, selective_stack, selected_mask


# ==============================================================================
# Loss Functions
# ==============================================================================

class SNRLoss(nn.Module):
    """
    Signal-to-Noise Ratio Loss for optimizing stacking weights.
    
    This loss function encourages the network to produce weights that maximize
    the SNR of the stacked signal within a specified time window.
    
    Parameters
    ----------
    tau : float
        Time lag parameter
    tmin : float
        Start time of signal window
    tmax : float
        End time of signal window
    delta : float, default=1.0
        Sampling interval
    eps : float, default=1e-8
        Small constant for numerical stability
    """
    
    def __init__(self, tau: float, tmin: float, tmax: float, 
                 delta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.tau_idx = int(tau / delta)
        self.tmin_idx = int(tmin / delta)
        self.tmax_idx = int(tmax / delta)
        self.delta = delta
        self.eps = eps
        
    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        """Compute RMS along the last dimension."""
        return torch.sqrt(torch.mean(x**2, dim=-1))
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute SNR loss.
        
        Parameters
        ----------
        data : torch.Tensor
            Stacked signal, shape [batch_size, sequence_length]
            
        Returns
        -------
        torch.Tensor
            SNR loss (mean of 1/SNR across batch)
        """
        length = data.size(1)
        zero_lag = length // 2
        
        # Extract symmetric component
        left = data[:, :zero_lag+1]
        right = data[:, zero_lag:]
        sym = torch.flip(left, dims=[1]) + right
        
        # Define signal and noise windows
        signal = sym[:, self.tmin_idx-self.tau_idx*2:self.tmax_idx+self.tau_idx*2]
        left_noise = sym[:, :self.tmin_idx-self.tau_idx*2]
        right_noise = sym[:, self.tmax_idx+self.tau_idx*2:]
        combined_noise = torch.cat([left_noise, right_noise], dim=-1)
        
        # Compute SNR
        snr = self._rms(signal) / (self._rms(combined_noise) + self.eps)
        
        # Return mean of inverse SNR (to minimize)
        return torch.mean(1. / snr)


class SymLoss(nn.Module):
    """
    Symmetry Loss for measuring anti-symmetric noise in stacked signals.
    
    This loss quantifies the asymmetry in the stacked signal, which should
    be symmetric around zero lag for ideal signals.
    
    Parameters
    ----------
    eps : float, default=1e-8
        Small constant for numerical stability
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    @staticmethod
    def _antisymmetry_noise(signal: torch.Tensor) -> torch.Tensor:
        """Compute anti-symmetric noise energy."""
        flipped = torch.flip(signal, dims=[-1])
        diff = signal - flipped
        return torch.sqrt(torch.mean(diff**2))
    
    @staticmethod
    def _signal_energy(signal: torch.Tensor) -> torch.Tensor:
        """Compute symmetric component energy."""
        length = signal.size(1)
        zero_lag = length // 2
        left = signal[:, :zero_lag+1]
        right = signal[:, zero_lag:]
        sym = torch.flip(left, dims=[1]) + right
        return torch.sqrt(torch.mean(sym**2, dim=-1))
    
    def forward(self, stacked_signal: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetry loss.
        
        Parameters
        ----------
        stacked_signal : torch.Tensor
            Stacked signal, shape [batch_size, sequence_length]
            
        Returns
        -------
        torch.Tensor
            Symmetry loss (ratio of anti-symmetric noise to signal energy)
        """
        S = self._antisymmetry_noise(stacked_signal)
        A_x_sq = self._signal_energy(stacked_signal)
        loss = S / (A_x_sq + self.eps)
        return loss.mean()


class stdLoss(nn.Module):
    """
    std-based Loss for encouraging diverse weight distributions.
    
    This loss encourages the network to use a diverse set of weights
    rather than concentrating on a few signals.
    
    Parameters
    ----------
    eps : float, default=1e-8
        Small constant for numerical stability
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute std loss.
        
        Parameters
        ----------
        weights : torch.Tensor
            Stacking weights, shape [batch_size, n_signals] or [n_signals]
            
        Returns
        -------
        torch.Tensor
            Negative standard deviation of weights (to maximize diversity)
        """
        return -torch.std(weights)


# ==============================================================================
# Neural Network Model
# ==============================================================================

class StackingNet(nn.Module):
    """
    Neural Network for intelligent signal stacking.
    
    This network learns optimal weights for stacking multiple signals
    to maximize signal quality according to specified criteria.
    
    Parameters
    ----------
    lag : float, default=500
        Maximum time lag (symmetric around zero)
    delta : float, default=1.0
        Sampling interval
    hidden_dims : List[int], optional
        Dimensions of hidden layers. Default: [512, 256, 128, 64, 16]
    """
    
    def __init__(self, lag: float = 500, delta: float = 1.0, 
                 hidden_dims: Optional[List[int]] = None):
        super().__init__()
        self.lag = lag
        self.delta = delta
        
        # Calculate input dimension
        input_dim = int(2 * lag / delta) + 1
        
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 16]
        
        # Build network layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using appropriate schemes."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signals, shape [n_signals, signal_length] or 
            [batch_size, n_signals, signal_length]
            
        Returns
        -------
        stacked_signal : torch.Tensor
            Weighted stack of input signals
        weights : torch.Tensor
            Learned weights for each input signal
        """
        # Ensure correct data type
        x = x.to(torch.float32)
        
        # Handle different input shapes
        if x.dim() == 2:
            # [n_signals, signal_length]
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
        elif x.dim() == 3:
            # [batch_size, n_signals, signal_length]
            batch_size = x.shape[1]
            x_flat = x.view(batch_size, -1)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        
        # Predict weights
        weights = self.net(x_flat)  # [batch_size, 1]
        
        # Weighted stacking
        # Note: Original code used weights.T @ x_flat, but this seems incorrect
        # We'll implement proper weighted stacking
        weighted_signals = x_flat * weights  # Broadcast weights
        stacked_signal = torch.sum(weighted_signals, dim=0, keepdim=True)
        
        return stacked_signal, weights


