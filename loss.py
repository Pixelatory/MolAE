import torch.nn.functional as F
import torch

def recon_loss(outputs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor = None, **kwargs):
    outputs = F.softmax(outputs, dim=-1)

    if weights is not None:
        # Masked language modelling
        outputs = outputs[weights==1]
        labels = labels[weights==1]
    else:
        # language modelling
        outputs = outputs.flatten(0, 1)
        labels = labels.flatten(0, 1)
    return F.cross_entropy(outputs, labels, **kwargs)

def vae_loss(outputs: torch.Tensor, labels: torch.Tensor, log_var: torch.Tensor, 
             mu: torch.Tensor, weights: torch.Tensor = None, kld_weight: float = 1.):
    r_loss = recon_loss(outputs, labels, weights)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1))
    return r_loss + kld_weight * kld_loss

def amim_loss(outputs: torch.Tensor, labels: torch.Tensor, z: torch.Tensor,
              weights: torch.Tensor = None, kld_weight: float = 1.):
    r_loss = recon_loss(outputs, labels, weights)
    kld_loss = 