"""Utility functions for HANNA model operations."""

import os
from typing import Optional

import torch

from models.HANNA.HANNA import HANNA_Ensemble_Multicomponent

# Constants
DEFAULT_NUM_MODELS = 10
DEFAULT_NODES = 96
DEFAULT_EMBEDDING_SIZE = 384


def load_ensemble(
    ensemble_path: str, 
    device: Optional[torch.device] = None, 
    num_models: int = DEFAULT_NUM_MODELS
) -> HANNA_Ensemble_Multicomponent:
    """Load HANNA ensemble model from checkpoint files.
    
    Args:
        ensemble_path: Path to directory containing ensemble model files.
        device: Device to load models on. If None, uses CPU.
        num_models: Number of models in the ensemble.
        
    Returns:
        Loaded ensemble model ready for inference.
        
    Raises:
        FileNotFoundError: If ensemble path doesn't exist.
        ValueError: If no model files found in ensemble path.
    """
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble path '{ensemble_path}' does not exist")
    
    if device is None:
        device = torch.device('cpu')
    
    # Generate model file paths
    model_paths = [
        os.path.join(ensemble_path, f'HANNA_parameters_binary{i}.pt') 
        for i in range(num_models)
    ]
    
    # Verify all model files exist
    missing_files = [path for path in model_paths if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            f"Missing ensemble model files: {missing_files}"
        )
    
    print(f'Loading ensemble with {num_models} models...')
    
    ensemble = HANNA_Ensemble_Multicomponent(
        model_paths=model_paths,
        Embedding_ChemBERT=DEFAULT_EMBEDDING_SIZE,
        nodes=DEFAULT_NODES,
        device=device,
    )
    ensemble.to(device)
    
    return ensemble