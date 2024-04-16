import math
from typing import List
import torch
import numpy as np

MASK_TOKEN = 0

def token_deletion(encoded_seq: List[int],
                   remove_percentage: float = None,
                   num_deleted_tokens: int = None,
                   generator: np.random.Generator = None) -> torch.Tensor:
    """
        From BART (https://arxiv.org/pdf/1910.13461.pdf):
        Random tokens are deleted from the input.
    """
    if remove_percentage is None and num_deleted_tokens is None:
        raise Exception("remove_percentage and num_deleted_tokens cannot both be None")
    
    if generator is None:
        generator = np.random.default_rng()

    if remove_percentage is not None:
        num_deleted_tokens = math.ceil(len(encoded_seq) * remove_percentage)

    remaining_tokens = len(encoded_seq) - num_deleted_tokens

    masked_seq = np.array(encoded_seq, copy=True)

    idxs = generator.permutation(np.arange(len(encoded_seq)))[:remaining_tokens]
    idxs = np.sort(idxs)
        
    return masked_seq[idxs]

def text_infilling(encoded_seq: List[int], 
                   mask_encoding_val: int,
                   mask_percentage: float = 0.15,
                   lam: int = 3,
                   generator: np.random.Generator = None) -> torch.Tensor:
    """
        From BART (https://arxiv.org/pdf/1910.13461.pdf):
        A number of text spans are sampled, with span lengths drawn 
        from a Poisson distribution. Each span is replaced with a 
        single mask token. 0-length spans correspond to the insertion 
        of mask tokens.
    """
    if generator is None:
        generator = np.random.default_rng()

    masked_sequence = encoded_seq.copy()

    max_mask_tokens = math.ceil(len(encoded_seq) * mask_percentage)
    masked_tokens = 0

    while masked_tokens < max_mask_tokens:
        start_index = generator.integers(0, len(masked_sequence))
        span_length = generator.poisson(lam)
        
        # TODO: fix mistake: start_index + span_length can be greater than length of list
        if span_length == 0:
            masked_sequence.insert(start_index, mask_encoding_val)
        else:
            masked_sequence[start_index : start_index + span_length + 1] = [mask_encoding_val]

        masked_tokens += span_length + 1
    
    return torch.tensor(masked_sequence, dtype=torch.int64)

res = token_deletion([i for i in range(100)], num_deleted_tokens=10)
print(res, len(res))
#res = text_infilling([5] * 100, MASK_TOKEN)