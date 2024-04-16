from typing import Tuple
import torch
from rdkit import Chem
import numpy as np
import inspect

SMI_REGEX_PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0-9])"


def create_causal_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    """
    Creates a causal mask to work with the Transformer model
    implementation (1 for masked out tokens)
    """
    return torch.triu(torch.ones(size=(batch_size, seq_len)), diagonal=1).int()


def create_pad_mask(x: torch.Tensor, pad_token_idx: int) -> torch.Tensor:
    """
    Creates a padding mask to work with the Transformer model
    implementation (1 for masked out tokens)
    """
    return (x == pad_token_idx).int()


def combine_masks(*masks: Tuple[torch.Tensor], method: str = 'union') -> torch.Tensor:
    """
    Combines various binary masks into a single tensor.

    Parameters:
    - method: Method of combining masks into a single tensor. Can be 'union', 'intersection', or 'xor'.
    """
    if method not in ['union', 'intersection', 'xor']:
        raise ValueError("Invalid method. Supported methods are 'union', 'intersection', 'xor'.")

    shapes = set(mask.shape for mask in masks)
    if len(shapes) != 1:
        raise ValueError("All masks must have the same shape.")
    
    combined_mask  = masks[0]

    for mask in masks[1:]:
        if method == 'union':
            combined_mask = torch.logical_or(combined_mask, mask)
        elif method == 'intersection':
            combined_mask = torch.logical_and(combined_mask, mask)
        elif method == 'xor':
            combined_mask = torch.logical_xor(combined_mask, mask)
    
    return combined_mask.int()


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        From: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py#L107

        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Parameters:
        - mu: Mean of the latent Gaussian (batch_size x d_model)
        - logvar: (Tensor) Standard deviation of the latent Gaussian (batch_size x d_model)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


def log_normal_diag(x, mu, log_var, average=False, dim=None):
    """
    From: https://github.com/seraphlabs-ca/MIM/blob/master/src/cond_pixelcnn/utils/distributions.py#L10
    """
    log_normal = -0.5 * (log_var + torch.pow(x - mu, 2) / torch.exp(log_var))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def randomize_smiles(smi: str, generator: np.random.Generator = None) -> str:
    m = Chem.MolFromSmiles(smi)
    ans = list(range(m.GetNumAtoms()))
    if generator is not None:
        generator.shuffle(ans)
    else:
        np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    smiles = Chem.MolToSmiles(nm, canonical=False)
    return smiles


def canonical_smiles(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    smiles = Chem.MolToSmiles(m, canonical=True)
    return smiles


def has_argument(fn: callable, arg_name: str) -> bool:
    """
    Check if a function has an argument with a specific name.
    """
    signature = inspect.signature(fn)
    return arg_name in signature.parameters