import torch


def gen_pos_encoding(seq_length, num_channels):
    """
    Generate positional encoding for a sequence.

    Parameters:
    - seq_length (int): The length of the sequence.
    - num_channels (int): The number of channels at each position.

    Returns:
    - pe (torch.Tensor): A (seq_length, num_channels) matrix containing the positional encoding.

    The positional encoding is calculated using the formula:
    - PE(t, 2i) = sin(t / 10000^(2i/num_channels))
    - PE(t, 2i+1) = cos(t / 10000^(2i/num_channels))
    """
    pe = torch.zeros(seq_length, num_channels)
    t = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    i = torch.arange(0, num_channels, 2, dtype=torch.float).unsqueeze(0)
    div_term = torch.pow(10000.0, (2 * i) / num_channels)
    pe[:, 0::2] = torch.sin(t / div_term)
    # Handle the case when num_channels is odd
    pe[:, 1::2] = torch.cos(t / div_term) if num_channels % 2 == 0 else torch.cos(t / div_term[:, :-1])
    return pe
