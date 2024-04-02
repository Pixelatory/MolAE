from typing import Tuple
import torch

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