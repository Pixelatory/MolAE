import torch.nn.functional as F
import torch

def recon_loss(outputs, labels, weights=None):
    if weights is not None:
        outputs = outputs[weights==1]
        labels = labels[weights==1]
    else:
        outputs = outputs.flatten(0, 1)
        labels = labels.flatten(0, 1)
    return F.cross_entropy(outputs, labels)