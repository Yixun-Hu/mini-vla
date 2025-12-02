#!/usr/bin/env python3
"""
Helper functions for training and inference.
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_data(data: torch.Tensor, stats: dict) -> torch.Tensor:
    """
    Normalize data to [-1, 1] using min-max normalization.
    
    Args:
        data: (*, D) tensor
        stats: dict with 'min' and 'max' tensors of shape (D,)
    
    Returns:
        normalized: (*, D) tensor in range [-1, 1]
    """
    min_val = stats['min'].to(data.device)
    max_val = stats['max'].to(data.device)
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_val = torch.where(range_val < 1e-7, torch.ones_like(range_val), range_val)
    
    # Normalize to [0, 1] then to [-1, 1]
    normalized = (data - min_val) / range_val
    normalized = normalized * 2.0 - 1.0
    
    return normalized


def denormalize_data(data: torch.Tensor, stats: dict) -> torch.Tensor:
    """
    Denormalize data from [-1, 1] back to original range.
    
    Args:
        data: (*, D) tensor in range [-1, 1]
        stats: dict with 'min' and 'max' tensors of shape (D,)
    
    Returns:
        denormalized: (*, D) tensor in original range
    """
    min_val = stats['min'].to(data.device)
    max_val = stats['max'].to(data.device)
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_val = torch.where(range_val < 1e-7, torch.ones_like(range_val), range_val)
    
    # Denormalize from [-1, 1] to [0, 1] then to original range
    normalized = (data + 1.0) / 2.0
    denormalized = normalized * range_val + min_val
    
    return denormalized


def compute_statistics(data: np.ndarray) -> dict:
    """
    Compute min and max statistics for normalization.
    
    Args:
        data: (N, D) numpy array
    
    Returns:
        stats: dict with 'min' and 'max' tensors
    """
    return {
        'min': torch.tensor(data.min(axis=0), dtype=torch.float32),
        'max': torch.tensor(data.max(axis=0), dtype=torch.float32),
    }