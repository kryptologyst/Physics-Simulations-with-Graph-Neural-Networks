"""Utility functions for device management and deterministic seeding."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    **kwargs
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save.
        path: Path to save the checkpoint.
        optimizer: Optimizer state (optional).
        epoch: Current epoch (optional).
        loss: Current loss (optional).
        **kwargs: Additional metadata to save.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        **kwargs
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if loss is not None:
        checkpoint["loss"] = loss
        
    torch.save(checkpoint, path)


def load_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> dict:
    """Load model checkpoint.
    
    Args:
        model: Model to load weights into.
        path: Path to the checkpoint.
        optimizer: Optimizer to load state into (optional).
        device: Device to load the checkpoint on.
        
    Returns:
        dict: Checkpoint metadata.
    """
    if device is None:
        device = get_device()
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    return checkpoint
