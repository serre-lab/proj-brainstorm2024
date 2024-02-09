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


def contrastive_loss(embeddings, movie_indices, margin=0.0):
    # Normalize embeddings to unit vectors.
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Calculate cosine similarity matrix.
    similarity_matrix = torch.matmul(embeddings, embeddings.T)

    # Create masks for positive and negative pairs.
    positive_mask = movie_indices.unsqueeze(1) == movie_indices.unsqueeze(0)
    negative_mask = ~positive_mask

    # Remove self-similarity by setting diagonal to zero.
    eye_mask = torch.eye(embeddings.size(0), device=embeddings.device).bool()
    positive_mask.masked_fill_(eye_mask, False)

    # Calculate positive loss (maximize similarity, i.e., minimize negative similarity).
    positive_loss = -similarity_matrix[positive_mask].mean()

    # Calculate negative loss (ensure similarity is below a threshold, i.e., below margin).
    negative_similarities = similarity_matrix[negative_mask]
    negative_loss = F.relu(negative_similarities - margin).mean()

    # Combine losses.
    loss = positive_loss + negative_loss
    return loss


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
