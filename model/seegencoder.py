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

    def __init__(self, num_heads=2, num_encoder_layers=6, dim_feedforward=2048):
        super().__init__()

        self.num_input_channels = 84
        self.num_output_channels = 128
        self.input_length = 5120
        self.output_length = 196

        # Positional encoding
        positional_encoding = gen_pos_encoding(self.input_length, self.num_input_channels)
        self.register_buffer('positional_encoding', positional_encoding)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_input_channels, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Length matching layer
        self.length_matching_layer = nn.Linear(self.input_length, self.output_length)

        # Channel matching layer
        self.channel_matching_layer = nn.Linear(self.num_input_channels, self.num_output_channels)

    def forward(self, x):
        """
        Parameters:
        - x (torch.Tensor): A (batch_size, num_input_channels, input_length) tensor containing the input sEEG data.

        Returns:
        - x (torch.Tensor): A (batch_size, output_length, num_output_channels) tensor containing the output
        sequence.
        """
        x = x.permute(0, 2, 1)

        x += self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.channel_matching_layer(x)
        x = x.permute(0, 2, 1)
        x = self.length_matching_layer(x)
        x = x.permute(0, 2, 1)
        return x


if __name__ == '__main__':
    import torch

    num_heads = 2
    num_encoder_layers = 6
    dim_feedforward = 2048

    model = SEEGEncoder(num_heads, num_encoder_layers, dim_feedforward)

    seegs = torch.randn(2, 84, 5120)

    output = model(seegs)

    assert output.shape == (2, 196, 128)
