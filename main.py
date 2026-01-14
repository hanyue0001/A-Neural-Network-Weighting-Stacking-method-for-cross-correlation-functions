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


# ==============================================================================
# Training Interface
# ==============================================================================

class NeuralStackingTrainer:
    """
    Trainer for the StackingNet neural network.
    
    Parameters
    ----------
    model : StackingNet
        The neural network model
    tau : float
        Time lag parameter for SNR loss
    tmin : float
        Start time of signal window for SNR loss
    tmax : float
        End time of signal window for SNR loss
    delta : float, default=1.0
        Sampling interval
    snr_weight : float, default=1.0
        Weight for SNR loss
    sym_weight : float, default=1.0
        Weight for symmetry loss
    std_weight : float, default=0.05
        Weight for std loss
    device : str, optional
        Device to use ('cuda' or 'cpu'). Auto-detected if None.
    """
    
    def __init__(self, model: StackingNet, tau: float, tmin: float, tmax: float,
                 delta: float = 1.0, snr_weight: float = 1.0, 
                 sym_weight: float = 1.0, std_weight: float = 0.05,
                 device: Optional[str] = None):
        
        self.model = model
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Loss functions
        self.snr_loss = SNRLoss(tau, tmin, tmax, delta).to(self.device)
        self.sym_loss = SymLoss().to(self.device)
        self.std_loss = stdLoss().to(self.device)
        
        # Loss weights
        self.snr_weight = snr_weight
        self.sym_weight = sym_weight
        self.std_weight = std_weight
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.history = {
            'total_loss': [],
            'snr_loss': [],
            'sym_loss': [],
            'std_loss': [],
            'learning_rate': []
        }
    
    def configure_optimizer(self, learning_rate: float = 0.001, 
                           weight_decay: float = 0.0):
        """
        Configure the optimizer and learning rate scheduler.
        
        Parameters
        ----------
        learning_rate : float, default=0.001
            Initial learning rate
        weight_decay : float, default=0.0
            L2 regularization weight
        """
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=50, 
            gamma=0.5
        )
    
    def train_epoch(self, data: torch.Tensor) -> dict:
        """
        Train for one epoch.
        
        Parameters
        ----------
        data : torch.Tensor
            Training data, shape [1, n_signals, signal_length]
            
        Returns
        -------
        dict
            Dictionary containing loss values for the epoch
        """
        self.model.train()
        
        # Move data to device
        data = data.to(self.device)
        
        # Forward pass
        stacked_signal, weights = self.model(data)
        
        # Compute losses
        loss_snr = self.snr_loss(stacked_signal)
        loss_sym = self.sym_loss(stacked_signal)
        loss_std = self.std_loss(weights)
        
        # Weighted total loss
        total_loss = (
            self.snr_weight * loss_snr + 
            self.sym_weight * loss_sym + 
            self.std_weight * loss_std
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Record losses
        epoch_losses = {
            'total_loss': total_loss.item(),
            'snr_loss': loss_snr.item(),
            'sym_loss': loss_sym.item(),
            'std_loss': loss_std.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return epoch_losses
    
    def train(self, data: np.ndarray, num_epochs: int, 
              learning_rate: float = 0.001, verbose: bool = True) -> dict:
        """
        Train the model on the entire dataset.
        
        Parameters
        ----------
        data : np.ndarray
            Training data, shape [n_signals, signal_length]
        num_epochs : int
            Number of training epochs
        learning_rate : float, default=0.001
            Initial learning rate
        verbose : bool, default=True
            Whether to print progress information
            
        Returns
        -------
        dict
            Training history
        """
        # Convert data to tensor and add batch dimension
        data_tensor = torch.from_numpy(data).unsqueeze(0).to(torch.float32)
        
        # Configure optimizer if not already configured
        if self.optimizer is None:
            self.configure_optimizer(learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_losses = self.train_epoch(data_tensor)
            
            # Record history
            for key, value in epoch_losses.items():
                self.history[key].append(value)
            
            # Print progress
            if verbose and (((epoch+1) % 50 == 0) or epoch == 0):
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Total Loss: {epoch_losses['total_loss']:.4f}, "
                      f"SNR Loss: {epoch_losses['snr_loss']:.4f}, "
                      f"Sym Loss: {epoch_losses['sym_loss']:.4f}")
        
        if verbose:
            print("Training complete!")
        
        return self.history
    
    def save_model(self, path: str):
        """
        Save the trained model.
        
        Parameters
        ----------
        path : str
            Path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'history': self.history,
            'loss_weights': {
                'snr': self.snr_weight,
                'sym': self.sym_weight,
                'std': self.std_weight
            }
        }, path)
    
    def load_model(self, path: str):
        """
        Load a trained model.
        
        Parameters
        ----------
        path : str
            Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Model loaded from {path}")


# ==============================================================================
# High-level API
# ==============================================================================

def create_stacking_network(lag: float = 500, delta: float = 1.0,
                           hidden_dims: Optional[List[int]] = None) -> StackingNet:
    """
    Create a stacking neural network.
    
    Parameters
    ----------
    lag : float, default=500
        Maximum time lag
    delta : float, default=1.0
        Sampling interval
    hidden_dims : List[int], optional
        Dimensions of hidden layers
        
    Returns
    -------
    StackingNet
        Initialized neural network
    """
    return StackingNet(lag=lag, delta=delta, hidden_dims=hidden_dims)


def train_stacking_network(data: np.ndarray, tau: float, tmin: float, tmax: float,
                          num_epochs: int = 500, lag: float = 500, delta: float = 1.0,
                          snr_weight: float = 1.0, sym_weight: float = 1.0,
                          std_weight: float = 0.05, learning_rate: float = 0.001,
                          device: Optional[str] = None, verbose: bool = True) -> Tuple[StackingNet, dict]:
    """
    High-level function to create and train a stacking network.
    
    Parameters
    ----------
    data : np.ndarray
        Input signals, shape [n_signals, signal_length]
    tau : float
        Time lag parameter
    tmin : float
        Start time of signal window
    tmax : float
        End time of signal window
    num_epochs : int, default=500
        Number of training epochs
    lag : float, default=500
        Maximum time lag
    delta : float, default=1.0
        Sampling interval
    snr_weight : float, default=1.0
        Weight for SNR loss
    sym_weight : float, default=1.0
        Weight for symmetry loss
    std_weight : float, default=0.05
        Weight for std loss
    learning_rate : float, default=0.001
        Learning rate
    device : str, optional
        Device to use ('cuda' or 'cpu')
    verbose : bool, default=True
        Whether to print training progress
        
    Returns
    -------
    StackingNet
        Trained neural network
    dict
        Training history
    """
    # Create model
    model = create_stacking_network(lag=lag, delta=delta)
    
    # Create trainer
    trainer = NeuralStackingTrainer(
        model=model,
        tau=tau,
        tmin=tmin,
        tmax=tmax,
        delta=delta,
        snr_weight=snr_weight,
        sym_weight=sym_weight,
        std_weight=std_weight,
        device=device
    )
    
    # Train model
    history = trainer.train(
        data=data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        verbose=verbose
    )
    
    return model, history


def apply_stacking_network(model: StackingNet, data: np.ndarray, 
                          device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a trained stacking network to new data.
    
    Parameters
    ----------
    model : StackingNet
        Trained neural network
    data : np.ndarray
        Input signals, shape [n_signals, signal_length]
    device : str, optional
        Device to use ('cuda' or 'cpu')
        
    Returns
    -------
    stacked_signal : np.ndarray
        Weighted stack of input signals
    weights : np.ndarray
        Learned weights for each input signal
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Convert data to tensor
    data_tensor = torch.from_numpy(data).unsqueeze(0).to(torch.float32).to(device)
    
    # Apply model
    with torch.no_grad():
        stacked_signal, weights = model(data_tensor)
    
    # Convert to numpy
    stacked_signal = stacked_signal.cpu().numpy().squeeze()
    weights = weights.cpu().numpy().squeeze()
    
    return stacked_signal, weights


# ==============================================================================
# Utility Functions
# ==============================================================================

def normalize_signals(signals: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Normalize signals by their maximum absolute value.
    
    Parameters
    ----------
    signals : np.ndarray
        Input signals
    axis : int, default=1
        Axis along which to compute maximum
        
    Returns
    -------
    np.ndarray
        Normalized signals
    """
    if signals.ndim == 1:
        return signals / np.abs(signals).max()
    else:
        return signals / np.abs(signals).max(axis=axis).reshape(-1, 1)


def calculate_snr_ratio(data: np.ndarray, tau: float, tmin: float, 
                       tmax: float, delta: float = 1.0) -> float:
    """
    Calculate SNR ratio using trailing noise window.
    
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
        SNR ratio
    """
    tau_idx = int(tau/delta)
    tmin_idx = int(tmin/delta)
    tmax_idx = int(tmax/delta)
    
    length = len(data)
    zero_lag = int(length/2)
    
    left = data[:zero_lag+1]
    right = data[zero_lag:]
    left = left[::-1]
    syme = np.sum(np.vstack((left, right)), 0)
    
    signal = syme[tmin_idx-2*tau_idx:tmax_idx+2*tau_idx]
    trail = syme[tmax_idx+4*tau_idx:tmax_idx+8*tau_idx]
    
    return rms(signal) / rms(trail)


# ==============================================================================
# Example Usage
# ==============================================================================

def example_usage():
    """Demonstrate basic usage of the neural stacking framework."""
    import matplotlib.pyplot as plt
    
    print("Neural Stacking Framework - Example Usage")
    print("=" * 50)
    
    # 1. Generate synthetic data
    n_signals = 100
    signal_length = 1001
    np.random.seed(42)
    
    # Create synthetic signals with varying noise levels
    clean_signal = np.exp(-np.linspace(-5, 5, signal_length)**2)
    signals = []
    
    for i in range(n_signals):
        noise_level = 0.1 + 0.5 * np.random.rand()
        noise = noise_level * np.random.randn(signal_length)
        signal = clean_signal + noise
        signals.append(signal)
    
    data = np.array(signals)
    
    print(f"Generated {n_signals} signals of length {signal_length}")
    print(f"Data shape: {data.shape}")
    
    # 2. Traditional stacking (for comparison)
    linear_stack = stack_all(data)
    print(f"\nTraditional linear stacking completed")
    
    # 3. Neural stacking
    tau = 10.0
    tmin = 400.0
    tmax = 600.0
    delta = 1.0
    
    print("\nTraining neural stacking network...")
    model, history = train_stacking_network(
        data=data,
        tau=tau,
        tmin=tmin,
        tmax=tmax,
        num_epochs=200,
        lag=500,
        delta=delta,
        snr_weight=1.0,
        sym_weight=1.0,
        std_weight=0.05,
        learning_rate=0.001,
        verbose=True
    )
    
    # 4. Apply trained model
    print("\nApplying trained model...")
    neural_stack, weights = apply_stacking_network(model, data)
    
    # 5. Compare results
    linear_snr = calculate_snr_ratio(linear_stack, tau, tmin, tmax, delta)
    neural_snr = calculate_snr_ratio(neural_stack, tau, tmin, tmax, delta)
    
    print(f"\nResults:")
    print(f"Linear stacking SNR ratio: {linear_snr:.4f}")
    print(f"Neural stacking SNR ratio: {neural_snr:.4f}")
    print(f"Improvement: {100 * (neural_snr - linear_snr) / linear_snr:.1f}%")
    
    # 6. Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    axes[0, 1].plot(history['snr_loss'], label='SNR Loss')
    axes[0, 1].plot(history['sym_loss'], label='Sym Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    axes[1, 0].plot(history['std_loss'])
    axes[1, 0].set_title('std Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    
    axes[1, 1].plot(history['learning_rate'])
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.show()
    
    # 7. Plot stacking results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    time = np.arange(-500, 501) * delta
    
    axes[0].plot(time, linear_stack, label='Linear Stack', alpha=0.8)
    axes[0].plot(time, neural_stack, label='Neural Stack', alpha=0.8)
    axes[0].axvspan(tmin-2*tau, tmax+2*tau, alpha=0.2, color='gray', label='Signal Window')
    axes[0].set_xlabel('Time Lag')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Stacking Results Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(range(len(weights)), weights.flatten())
    axes[1].set_xlabel('Signal Index')
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Learned Stacking Weights')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model, history, linear_stack, neural_stack, weights


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    # Run example if executed directly
    example_usage()