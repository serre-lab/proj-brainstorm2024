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
