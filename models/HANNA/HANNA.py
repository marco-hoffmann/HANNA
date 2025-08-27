import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import  combinations
import os
from transformers import AutoTokenizer, AutoModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from rdkit import Chem

class LipschitzLinearLayer(nn.Module):
    """
    A linear layer that applies spectral normalization by power iteration,
    with additional scaling by softplus(ci). In training mode, it updates 
    the spectral norm estimates; in eval mode it may skip or reuse them.
    
    Introduced: A warmstart mechanism to skip normalization for initial epochs.
    During warmstart, the layer bypasses normalization, which can speed up early training.
    """

    def __init__(self, in_features, out_features, bias=True,
                 n_power_iterations=2):
        super().__init__()
        # Standard linear parameter initialization
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize trainable scale factor ci (used when warmup is skipped!)
        ci_init = 4.0
        self.ci = nn.Parameter(torch.tensor(ci_init, dtype=torch.float32))
        self.ci.requires_grad = True

        # Number of power iterations for refining singular value estimates
        self.n_power_iterations = n_power_iterations

        # For debugging: store intermediate normalized and scaled weights
        self._W_normed = None
        self._W_scaled = None
        # Small epsilon value to avoid division by zero 
        self.eps = 1e-12

        # Initialize buffers for power iteration: _u and _v approximating leading singular vectors
        out_f = out_features
        in_f = in_features
        u_init = F.normalize(torch.randn(out_f), dim=0)
        v_init = F.normalize(torch.randn(in_f), dim=0)
        self.register_buffer("_u", u_init)
        self.register_buffer("_v", v_init)

        # Do one power iteration to initialize the buffers
        W_raw = self.linear.weight
        self._power_iteration(W_raw)

        # Detach and clone _u and _v to avoid in-place modification issues during gradient computation
        u_detached = self._u.clone().detach()
        v_detached = self._v.clone().detach()

        # Compute the approximate largest singular value using detached vectors
        largest_sv = u_detached.dot(torch.mv(W_raw, v_detached))

        # Normalize weight to enforce spectral norm ~1 (Eq. (22))
        W_normed = W_raw / (largest_sv + self.eps)

        # Scale by softplus(ci) to adjust Lipschitz constant (Eq. (9))
        softplus_ci = F.softplus(self.ci)
        W_scaled = W_normed * softplus_ci

        # Store values for inspection/debugging
        self._W_normed = W_normed.detach().clone()
        self._W_scaled = W_scaled.detach().clone()

    def _power_iteration(self, W):
        """
        Perform power iteration to update buffers _u and _v.
        According to https://arxiv.org/pdf/1802.05957 starting on page 15:
          - v <- (W^T u) / ||W^T u||  (Eq. (20))
          - u <- (W v) / ||W v||      (Eq. (21))
        Repeating for self.n_power_iterations approximates the largest singular value:
          largest_sv ~ u^T W v (Eq. (19)).
        """
        with torch.no_grad():
            u = self._u
            v = self._v
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(W.transpose(0, 1), u), dim=0)
                u = F.normalize(torch.mv(W, v), dim=0)
            self._u.copy_(u)
            self._v.copy_(v)
    def forward(self, x):
        """
        Forward pass with optional warmstart.
        
        - During warmstart: bypasses spectral normalization, performing a standard linear transformation.
        - Otherwise, in training mode: updates _u and _v using power iteration to approximate the largest singular value.
        - Computes largest_sv from detached buffers to safely perform normalization.
        - Normalizes weight: W_normed = W_raw / largest_sv, ensuring ||W_normed||_2 <= 1.
        - Scales the normalized weight by softplus(ci) to enforce desired Lipschitz constant.
        - Returns the linear transformation using the final scaled weight.
        """

        W_raw = self.linear.weight

        # Update spectral norm estimates only during training mode
        if self.training:
            self._power_iteration(W_raw)

        # Detach and clone _u and _v to avoid in-place modification issues during gradient computation
        u_detached = self._u.clone().detach()
        v_detached = self._v.clone().detach()

        # Compute the approximate largest singular value using detached vectors
        largest_sv = u_detached.dot(torch.mv(W_raw, v_detached))

        # Normalize weight to enforce spectral norm ~1 (Eq. (22))
        W_normed = W_raw / (largest_sv + self.eps)

        # Scale by softplus(ci) to adjust Lipschitz constant (Eq. (9))
        softplus_ci = F.softplus(self.ci)
        W_scaled = W_normed * softplus_ci

        # Store values for inspection/debugging
        self._W_normed = W_normed.detach().clone()
        self._W_scaled = W_scaled.detach().clone()

        # Perform linear transformation using the final scaled weight
        return F.linear(x, W_scaled, self.linear.bias)
   
class HANNA_Ensemble_Multicomponent(nn.Module):
    def __init__(self, model_paths, Embedding_ChemBERT=384, nodes=96, device=None):
        super(HANNA_Ensemble_Multicomponent, self).__init__()
        self.models = nn.ModuleList()
        self.device = device if device else torch.device('cpu')

        for path in model_paths:
            # Load the model state dict
            model_state = torch.load(path, map_location=self.device)
            # Initialize a new model instance with required arguments
            model = HANNA_Multicomponent(Embedding_ChemBERT=Embedding_ChemBERT, nodes=nodes)
            # Load the state dict into the model
            model.load_state_dict(model_state, strict=False)
            # Move model to the device
            model.to(self.device)
            # Set the model to evaluation mode
            model.eval()
            # Append to the list of models
            self.models.append(model)

    def forward(self, temperature, mole_fractions, E_i):
        ln_gammas_list = []
        gE_list = []

        # Move inputs to the device
        temperature = temperature.to(self.device)
        mole_fractions = mole_fractions.to(self.device)
        E_i = E_i.to(self.device)

        for model in self.models:
            # Do not use torch.no_grad(), since gradients are required in HANNA's forward
            ln_gammas, gE = model(temperature, mole_fractions, E_i)
            ln_gammas_list.append(ln_gammas)
            gE_list.append(gE)

        # Stack the outputs and compute the mean
        ln_gammas_mean = torch.mean(torch.stack(ln_gammas_list), dim=0)
        gE_mean = torch.mean(torch.stack(gE_list), dim=0)

        return ln_gammas_mean, gE_mean
  
class HANNA_Multicomponent(nn.Module):
    def __init__(self, Embedding_ChemBERT=384, nodes=96, gamma=100):
        super(HANNA_Multicomponent, self).__init__()

        self.Embedding_ChemBERT = Embedding_ChemBERT  # Embedding size
        self.nodes = nodes  # Number of nodes in hidden layers
        self.gamma = gamma  # Fixed gamma parameter for RBF kernel

        # Component Embedding Network f_theta: Input E_i, Output theta(E_i)
        self.theta = nn.Sequential(
            LipschitzLinearLayer(self.Embedding_ChemBERT, self.nodes),
            nn.SiLU(),
        )

        # Mixture Embedding Network f_alpha:
        # Input: concatenation of [theta_E_i, xi_new, T]
        # Input size: nodes + 2
        self.alpha = nn.Sequential(
            LipschitzLinearLayer(self.nodes + 2, self.nodes),
            nn.SiLU(),
            LipschitzLinearLayer(self.nodes, self.nodes),
            nn.SiLU(),
        )

        # Property Network f_phi:
        # Input: output from alpha network (after summing over components)
        # Output: gE_NN_{i,j}
        self.phi = nn.Sequential(
            LipschitzLinearLayer(self.nodes, self.nodes),
            nn.SiLU(),
            LipschitzLinearLayer(self.nodes, 1)
        )

    def rbf_similarity(self, x, y):
        """
        Compute the RBF similarity between two tensors using a fixed gamma.
        x: Tensor of shape [..., D]
        y: Tensor of shape [..., D]
        Returns:
            RBF similarity tensor of shape [...]
        """
        # Compute squared Euclidean distance
        squared_diff = (x - y) ** 2
        squared_distance = squared_diff.sum(dim=-1)  # Sum over feature dimension

        # Apply RBF formula: exp(-gamma * squared_distance)
        rbf_sim = torch.exp(-self.gamma * squared_distance)
        return rbf_sim

    def forward(self, temperature, mole_fractions, E_i):
        # Determine batch_size (B) and number of components (N)
        batch_size, num_components, _ = E_i.shape  # E_i: [B, N, E]

        # Ensure temperature and mole_fractions have correct shapes
        temperature = temperature.view(batch_size, 1)  # Shape: [B, 1]
        mole_fractions = mole_fractions.view(batch_size, num_components - 1)  # Shape: [B, N-1]

        # Enable gradient tracking
        E_i.requires_grad_(True)
        temperature.requires_grad_(True)
        mole_fractions.requires_grad_(True)

        # Calculate mole fraction for the N-th component
        mole_fraction_N = 1 - mole_fractions.sum(dim=1, keepdim=True)  # Shape: [B, 1]
        mole_fractions_complete = torch.cat([mole_fractions, mole_fraction_N], dim=1)  # [B, N]

        # Fine-tuning of the component embeddings
        theta_E_i = self.theta(E_i)  # [B, N, nodes]

        # Compute RBF similarities between all components
        theta_E_i_expanded_i = theta_E_i.unsqueeze(2)  # [B, N, 1, nodes]
        theta_E_i_expanded_j = theta_E_i.unsqueeze(1)  # [B, 1, N, nodes]

        # Calculate RBF similarity matrix for each sample in the batch
        rbf_sim_matrix = self.rbf_similarity(theta_E_i_expanded_i, theta_E_i_expanded_j)  # [B, N, N]

        # Adjust mole fractions based on RBF similarities
        mole_fractions_expanded = mole_fractions_complete.unsqueeze(1)  # [B, 1, N]
        adjustment_terms = mole_fractions_expanded * rbf_sim_matrix  # [B, N, N]
        x_i_adjusted = adjustment_terms.sum(dim=2)  # [B, N]

        # Generate all unique pairs (i, j) with i < j
        pairs = list(combinations(range(num_components), 2))
        idx_i, idx_j = zip(*pairs)
        idx_i = torch.tensor(idx_i, device=E_i.device)
        idx_j = torch.tensor(idx_j, device=E_i.device)
        K = len(idx_i)  # number of unique pairs

        # Extract adjusted mole fractions for components i and j
        x_i_adjusted_i = x_i_adjusted[:, idx_i]  # [B, K]
        x_i_adjusted_j = x_i_adjusted[:, idx_j]  # [B, K]

        # Muggianu projection for each pair (i, j)
        # X_i^{ij} = (1 + x_i_adjusted_i - x_i_adjusted_j) / 2
        Xi_ij = (1 + x_i_adjusted_i - x_i_adjusted_j) / 2  # [B, K]
        Xj_ij = (1 + x_i_adjusted_j - x_i_adjusted_i) / 2  # [B, K]

        # Expand temperature to match pairs
        T_expanded = temperature.unsqueeze(1).expand(-1, K, -1)   # [B, K, 1]

        # Build c_i_new and c_j_new for each pair
        theta_E_i_paired = theta_E_i[:, idx_i, :]  # [B, K, nodes]
        theta_E_j_paired = theta_E_i[:, idx_j, :]  # [B, K, nodes]

        Xi_ij_expanded = Xi_ij.unsqueeze(-1)  # [B, K, 1]
        Xj_ij_expanded = Xj_ij.unsqueeze(-1)  # [B, K, 1]

        c_i_new = torch.cat([
            theta_E_i_paired,   # [B, K, nodes]
            Xi_ij_expanded,    # [B, K, 1]
            T_expanded          # [B, K, 1]
        ], dim=-1)  # [B, K, nodes + 2]

        c_j_new = torch.cat([
            theta_E_j_paired,   # [B, K, nodes]
            Xj_ij_expanded,    # [B, K, 1]
            T_expanded          # [B, K, 1]
        ], dim=-1)  # [B, K, nodes + 2]

        # Pass c_i_new and c_j_new through alpha network
        alpha_c_i_new = self.alpha(c_i_new)  # [B, K, nodes]
        alpha_c_j_new = self.alpha(c_j_new)  # [B, K, nodes]

        # Sum the alpha outputs to get combined representation
        alpha_c_ij = alpha_c_i_new + alpha_c_j_new  # [B, K, nodes]

        # Pass alpha_c_ij through phi to get gE_NN_{i,j}
        gE_NN_ij = self.phi(alpha_c_ij).squeeze(-1)  # [B, K]

        # RBF-based distance measure
        rbf_ij = rbf_sim_matrix[:, idx_i, idx_j]  # [B, K]

        # Compute correction factors using RBF-based distance (1 - RBF)
        x_i_paired = mole_fractions_complete[:, idx_i]  # [B, K]
        x_j_paired = mole_fractions_complete[:, idx_j]  # [B, K]
        correction_factor = x_i_paired * x_j_paired * (1 - rbf_ij)  # [B, K]

        # Compute adjusted gE_{i,j} for unique pairs (i, j)
        gE_ij = gE_NN_ij * correction_factor  # [B, K]

        # Sum contributions from all unique pairs (i, j)
        gE_binary = gE_ij.sum(dim=1)  # [B]

        # Initialize total gE with binary contributions
        gE = gE_binary.clone()  # [B]

        # Compute derivative of gE with respect to mole_fractions
        dgE_dxi = torch.autograd.grad(
            outputs=gE,
            inputs=mole_fractions,
            grad_outputs=torch.ones_like(gE),
            create_graph=True,
            retain_graph=True
        )[0]  # [B, N-1]

        # Compute sum_xj_dgE_dxj
        sum_xj_dgE_dxj = torch.sum(
            mole_fractions * dgE_dxi,
            dim=1,
            keepdim=True
        )  # [B, 1]

        # Calculate ln_gamma_i for i = 1, ..., N-1
        ln_gamma_i = gE.unsqueeze(1) + dgE_dxi - sum_xj_dgE_dxj  # [B, N-1]

        # Calculate ln_gamma_N
        ln_gamma_N = gE - sum_xj_dgE_dxj.squeeze(1)  # [B]

        # Concatenate the ln_gamma values for all components
        ln_gammas = torch.cat([ln_gamma_i, ln_gamma_N.unsqueeze(1)], dim=1)  # [B, N]

        return ln_gammas, gE

def initialize_ChemBERTA(model_name="DeepChem/ChemBERTa-77M-MTR", device=None):
       
    # Define ChemBERTa model and move it to the specified device
    #ChemBERTA = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name).to(device) # this loads the model from the internet
    ChemBERTA = AutoModel.from_pretrained(
        "models/ChemBERTa",
        local_files_only=True
    ).to(device)

    # Check if the vocabulary file already exists
    vocab_path = 'models/ChemBERTa/vocab.json'
    if not os.path.exists(vocab_path):
        # Save the tokenizer's vocabulary to the specified folder
        # Create the directory if it doesn't exist
        #tokenizer = AutoTokenizer.from_pretrained(model_name) # this loads the model from the internet
        tokenizer = AutoTokenizer.from_pretrained('models/ChemBERTa/', local_files_only=True)
        tokenizer.save_vocabulary('models/ChemBERTa/')
        
    # Load custom tokenizer using the saved vocab.json
    custom_tokenizer = Tokenizer(
        WordLevel.from_file(
            vocab_path,  # Path to your custom vocabulary file
            unk_token='[UNK]'
        )
    )

    # Set the pre-tokenizer to split SMILES characters (including handling Br, Cl, etc.)
    pre_tokenizer = Split(
        pattern=Regex(r"\[(.*?)\]|Br|Cl|."),
        behavior='isolated'
    )
    custom_tokenizer.pre_tokenizer = pre_tokenizer
    return ChemBERTA, custom_tokenizer

def get_smiles_embedding(smiles, custom_tokenizer, ChemBERTA, device, max_length=512):

    # canonicalize the SMILES
    smiles = canonicalize_smiles(smiles)

    # Tokenize the SMILES using your custom tokenizer
    custom_encoded = custom_tokenizer.encode(smiles)

    # Add [CLS] and [SEP] tokens
    CLS_token_id = 12  # Assuming 12 is the token ID for [CLS]
    SEP_token_id = 13  # Assuming 13 is the token ID for [SEP]
    PAD_token_id = 0   # Assuming 0 is the token ID for [PAD]

    # Create input_ids with [CLS] at the start and [SEP] at the end
    input_ids = [CLS_token_id] + custom_encoded.ids + [SEP_token_id]

    # Apply truncation if input_ids are longer than max_length
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length - 1] + [SEP_token_id]  # Ensure the sequence ends with [SEP]

    # Apply padding if input_ids are shorter than max_length
    if len(input_ids) < max_length:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [PAD_token_id] * padding_length

    # Convert the input_ids to PyTorch tensor format
    input_ids_tensor = torch.tensor([input_ids]).to(device)

    # Prepare attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids_tensor != PAD_token_id).long()

    with torch.no_grad():
        # Get embeddings from ChemBERTA
        emb = ChemBERTA(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask
        )["last_hidden_state"][:, 0, :].cpu().numpy()# Take [CLS] token embedding
    return emb

def canonicalize_smiles(smiles):
    # Canonicalize the SMILES
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return smiles