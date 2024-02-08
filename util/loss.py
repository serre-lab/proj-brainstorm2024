import torch
import torch.nn as nn
import torch.nn.functional as F


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


def general_contrast_loss(embeddings, movie_indices, temperature=0.07):

    """
    Calculate the contrastive loss using cos similarity for a batch of embeddings.
    Args:
    - embeddings (torch.Tensor): A (batch_size, feature_dimension)-sized tensor containing the embeddings.
    - movie_indices (torch.LongTensor): A (batch_size,)-sized tensor containing the movie index for each embedding.
    - temperature (float): The temperature for the softmax function.

    Returns:
    - c_loss (torch.Tensor): A scalar tensor containing the contrastive loss.
    """

    # Calculate cosine similarity matrix
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim = torch.matmul(embeddings, embeddings.T) / temperature
    sim_softmax = F.softmax(sim, dim=1)

    # Mask for positive and negative pairs
    positive_mask = movie_indices.unsqueeze(1) == movie_indices.unsqueeze(0)
    negative_mask = ~positive_mask

    # Mask for excluding self-similarity
    positive_mask.masked_fill_(torch.eye(embeddings.size(0), device=embeddings.device).bool(),
                               torch.tensor(0.0, device=embeddings.device))

    # Calculate the sim_softmax for negative pairs
    sim_softmax_negative = sim_softmax * negative_mask.float()
    sim_softmax_negative = sim_softmax_negative.sum(dim=1)/negative_mask.float().sum(dim=1)

    # loss is mean of the negative sim_softmax
    c_loss = sim_softmax_negative.mean()

    return c_loss


# def contrastive_loss(embeddings, movie_indices, margin=1.0):
#     """
#     Calculate the contrastive loss using Euclidean distance for a batch of embeddings.
#
#     Parameters:
#     - embeddings: Tensor of shape (batch_size, feature_dimension) containing the embeddings.
#     - movie_indices: LongTensor of shape (batch_size,) containing the movie index for each embedding.
#     - margin: The margin for separating dissimilar pairs. Lower cosine similarity for negative pairs is encouraged.
#
#     Returns:
#     - The contrastive loss for the batch based on cosine similarity.
#     """
#
#     # Calculate cosine similarity matrix
#     distance_matrix = torch.cdist(embeddings, embeddings, p=2)
#     distance_matrix.to(embeddings.device)
#
#     positive_mask = movie_indices.unsqueeze(1) == movie_indices.unsqueeze(0)
#     positive_mask.to(embeddings.device)
#     negative_mask = ~positive_mask
#
#     # Mask for excluding self-similarity
#     eye_mask = torch.eye(embeddings.size(0), device=embeddings.device).bool()
#
#     positive_mask.masked_fill_(eye_mask, torch.tensor(0.0, device=embeddings.device))
#
#     # Cosine similarity for positive and negative pairs
#     positive_sim = distance_matrix * positive_mask.float()
#     negative_sim = distance_matrix * negative_mask.float()
#
#     # For positive pairs, maximize cosine similarity
#     positive_loss = -positive_sim.sum() / positive_mask.float().sum()
#     # For negative pairs, minimize cosine similarity, using margin
#     negative_loss = F.relu(negative_sim - margin).sum() / negative_mask.float().sum()
#
#     # Combine losses
#     loss = positive_loss + negative_loss
#     return loss


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
