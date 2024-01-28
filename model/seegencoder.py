import torch.nn as nn
from util.model import gen_pos_encoding


class SEEGEncoder(nn.Module):
    """
    A Transformer Encoder for sEEG data.

    Parameters:
    - num_input_channels (int): The number of input channels in the sEEG data.
    - num_output_channels (int): The number of output channels.
    - input_length (int): The length of the padded input sequence.
    - output_length (int): The length of the output sequence.
    - num_heads (int): The number of heads in the multi-head attention.
    - num_encoder_layers (int): The number of encoder layers in the transformer.
    - dim_feedforward (int): The dimension of the feedforward network in the transformer.
    """

    def __init__(self, max_num_input_channels=100, num_output_channels=768, input_length=2560, output_length=1568,
                 num_heads=2, num_encoder_layers=6, dim_feedforward=2048):
        super().__init__()

        # Positional encoding
        positional_encoding = gen_pos_encoding(max_num_input_channels, input_length)
        self.register_buffer('positional_encoding', positional_encoding)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_length, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Length matching layer
        self.length_matching_layer = nn.Linear(input_length, output_length)

        # Channel matching layer
        self.channel_matching_layer = nn.Linear(max_num_input_channels, num_output_channels)

    def forward(self, x, padding_mask):
        """
        Parameters:
        - x (torch.Tensor): A (batch_size, max_num_input_channels, input_length) tensor containing the input sEEG data.
        - padding_mask (torch.Tensor): A (batch_size, max_num_input_channels) boolean tensor containing the mask for the
        padding.
        True indicates a padding position and False indicates a valid data position.

        Returns:
        - output (torch.Tensor): A (batch_size, output_length, num_output_channels) tensor containing the output
        sequence.
        """
        x += self.positional_encoding
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        output = self.length_matching_layer(output)
        output = output.permute(0, 2, 1)
        output = self.channel_matching_layer(output)

        return output


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    max_num_input_channels = 100
    num_output_channels = 768
    input_length = 2560
    output_length = 1568
    num_heads = 2
    num_encoder_layers = 6
    dim_feedforward = 2048

    model = SEEGEncoder(max_num_input_channels, num_output_channels, input_length, output_length, num_heads,
                        num_encoder_layers, dim_feedforward)

    seeg1 = torch.randn(85, input_length)
    padded_seeg1 = F.pad(seeg1, (0, 0, max_num_input_channels - 85, 0), 'constant', 0)
    padding_mask1 = torch.zeros(max_num_input_channels, dtype=torch.bool)
    padding_mask1[85:] = True

    seeg2 = torch.randn(100, input_length)
    padded_seeg2 = F.pad(seeg2, (0, 0, max_num_input_channels - 100, 0), 'constant', 0)
    padding_mask2 = torch.zeros(max_num_input_channels, dtype=torch.bool)

    seegs = torch.stack([padded_seeg1, padded_seeg2], dim=0)
    padding_masks = torch.stack([padding_mask1, padding_mask2], dim=0)

    output = model(seegs, padding_masks)

    assert output.shape == (2, output_length, num_output_channels)
