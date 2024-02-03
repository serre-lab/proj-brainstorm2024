import torch
import torch.nn as nn


def recon_loss(x, x_recon):
    """
    Reconstruction loss

    Parameters:
    - x (torch.Tensor): A (batch_size, ...)-sized tensor containing the input data.
    - x_recon (torch.Tensor): A (batch_size, ...)-sized tensor containing the reconstructed data.

    Returns:
    - loss (torch.Tensor): A scalar tensor containing the reconstruction loss.
    """
    return nn.MSELoss()(x, x_recon)


def general_contrast_loss(sim, label):
    # TODO: Implement the general contrastive loss
    return torch.tensor(0)


def agg_loss(r_loss, c_loss, alpha=1.0):
    """
    Aggregate loss

    Parameters:
    - r_loss (torch.Tensor): A scalar tensor containing the reconstruction loss.
    - c_loss (torch.Tensor): A scalar tensor containing the contrastive loss.
    - alpha (float): A scalar weight for the contrastive loss.

    Returns:
    - loss (torch.Tensor): A scalar tensor containing the aggregated loss.
    """
    return r_loss + alpha * c_loss
