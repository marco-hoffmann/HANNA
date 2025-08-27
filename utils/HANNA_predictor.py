
"""HANNA Predictor module for liquid-liquid equilibrium predictions."""

import torch
import numpy as np
import pandas as pd
from typing import Union, List

from models.HANNA.HANNA import get_smiles_embedding, initialize_ChemBERTA
from utils.utils import load_ensemble

import contextlib
import io
import sys

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

# Constants
MAX_LENGTH = 512
DEFAULT_ENSEMBLE_PATH = 'models/HANNA/ensemble'
TEMPERATURE_SCALER_PATH = 'utils/scalers/temperature_scaler.pkl'
BERT_SCALER_PATH = 'utils/scalers/bert_scaler.pkl'

class HANNA_Predictor:
    """HANNA predictor for liquid-liquid equilibrium predictions.
    
    This class provides an interface for predicting activity coefficients
    and excess Gibbs energy using the HANNA ensemble model.
    """
    
    def __init__(self, ensemble_path: str = DEFAULT_ENSEMBLE_PATH) -> None:
        """Initialize the HANNA predictor.
        
        Args:
            ensemble_path: Path to the ensemble models directory.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = load_ensemble(ensemble_path=ensemble_path, device=self.device)
        
        self.t_scaler = pd.read_pickle(TEMPERATURE_SCALER_PATH)
        self.bert_scaler = pd.read_pickle(BERT_SCALER_PATH)

        with nostdout():   
            self.ChemBERTA, self.tokenizer = initialize_ChemBERTA(device=self.device)
        
        # Validate scalers
        if self.t_scaler is None:
            raise ValueError(f'Failed to load temperature scaler from {TEMPERATURE_SCALER_PATH}')
        if self.bert_scaler is None:
            raise ValueError(f'Failed to load BERT scaler from {BERT_SCALER_PATH}')

    def _get_scaled_embeddings_from_smiles(self, smiles_list: List[str]) -> List[torch.Tensor]:
        """Get scaled molecular embeddings from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings.
            
        Returns:
            List of scaled embedding tensors.
            
        Raises:
            ValueError: If BERT scaler is not available.
        """
        if self.bert_scaler is None:
            raise ValueError('BERT scaler is not available.')

        embeddings = [
            get_smiles_embedding(
                smiles, self.tokenizer, self.ChemBERTA, 
                self.device, max_length=MAX_LENGTH
            ) for smiles in smiles_list
        ]
        
        scaled_embeddings = [
            torch.FloatTensor(self.bert_scaler.transform(embedding)).squeeze().to(self.device) 
            for embedding in embeddings
        ]

        return scaled_embeddings
    
    def _get_scaled_temperature(self, temperature: float) -> torch.Tensor:
        """Get scaled temperature tensor.
        
        Args:
            temperature: Temperature value in Kelvin.
            
        Returns:
            Scaled temperature tensor.
            
        Raises:
            ValueError: If temperature scaler is not available.
        """
        if self.t_scaler is None:
            raise ValueError('Temperature scaler is not available.')
        
        scaled_T = torch.FloatTensor(
            self.t_scaler.transform(np.array([[temperature]]))
        ).to(self.device)
        return scaled_T
    
    def predict(
        self, 
        smiles_list: List[str], 
        molar_fractions_all: Union[List[List[float]], np.ndarray], 
        temperature: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict activity coefficients and excess Gibbs energy.
        
        Args:
            smiles_list: List of SMILES strings for each component.
            molar_fractions_all: Molar fractions for each mixture composition.
                                Each row should sum to 1.0.
            temperature: Temperature in Kelvin.
            
        Returns:
            Tuple of (ln_gammas, gE) as numpy arrays.
            
        Raises:
            ValueError: If molar fractions don't sum to 1 or have invalid dimensions.
        """
        # Convert to numpy array if needed
        if isinstance(molar_fractions_all, list):
            molar_fractions_all = np.array(molar_fractions_all)

        # Validate molar fractions
        if not np.allclose(np.sum(molar_fractions_all, axis=1), 1.0, rtol=1e-5):
            raise ValueError('The sum of molar fractions must be close to 1.0')
        
        if molar_fractions_all.shape[1] != len(smiles_list):
            raise ValueError(
                f'Number of components in molar_fractions_all ({molar_fractions_all.shape[1]}) '
                f'must match number of SMILES ({len(smiles_list)})'
            )

        # Get scaled inputs
        scaled_T = self._get_scaled_temperature(temperature)
        scaled_embeddings = self._get_scaled_embeddings_from_smiles(smiles_list)

        # Prepare tensors for model
        batch_size = len(molar_fractions_all)
        temperature_tensor = scaled_T.repeat(batch_size, 1).to(self.device)
        x_values_tensor = torch.FloatTensor(molar_fractions_all[:, :-1]).to(self.device)
        embedding_tensor = (
            torch.stack(scaled_embeddings)
            .repeat(batch_size, 1, 1)
            .to(self.device)
        )

        # Make prediction
        ln_gammas, gE = self.model(temperature_tensor, x_values_tensor, embedding_tensor)

        return ln_gammas.detach().cpu().numpy(), gE.detach().cpu().numpy()