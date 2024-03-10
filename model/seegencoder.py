import torch
import torch.nn as nn
from util.model import gen_pos_encoding


class SEEGEncoder(nn.Module):
    """
    A Transformer Encoder for sEEG data.
    Parameters:
    - num_heads (int): The number of heads in the multi-head attention.
    - num_encoder_layers (int): The number of encoder layers in the transformer.
    - dim_feedforward (int): The dimension of the feedforward network in the transformer.
    - num_input_channels (int): The number of input channels in the sEEG data.
    - input_length (int): The length of the padded input sequence.
    """
    def __init__(self, num_heads=2, num_encoder_layers=6, dim_feedforward=2048,
                 num_input_channels=84, input_length=5120):
        super().__init__()
        output_length = 1
        num_output_channels = 768

        # Positional encoding
        positional_encoding = gen_pos_encoding(input_length, num_input_channels)
        self.register_buffer('positional_encoding', positional_encoding)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_input_channels, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.length_matching_layer = nn.Linear(input_length, output_length)

        self.channel_matching_layer = nn.Linear(num_input_channels, num_output_channels)

    def forward(self, x):
        """
        Parameters:
        - x (torch.Tensor): A (batch_size, num_input_channels, input_length) tensor containing the input sEEG data.
        Returns:
        - x (torch.Tensor): A (batch_size, num_output_channels) tensor containing the sEEG embedding.
        sequence.
        """
        x = x.permute(0, 2, 1)
        x += self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.length_matching_layer(x)
        x = x.squeeze(-1)
        x = self.channel_matching_layer(x)
        return x


class SEEGEncoderProj(nn.Module):
    """
    A Transformer Encoder for sEEG data.
    Parameters:
    - num_heads (int): The number of heads in the multi-head attention.
    - num_encoder_layers (int): The number of encoder layers in the transformer.
    - dim_feedforward (int): The dimension of the feedforward network in the transformer.
    - c (int): The number of time steps to group together.
    - num_input_channels (int): The number of input channels in the sEEG data.
    - input_length (int): The length of the padded input sequence.
    """
    def __init__(self, num_heads=6, num_encoder_layers=6, dim_feedforward=2048, c=10,
                 num_input_channels=84, input_length=5120):
        super().__init__()

        num_output_channels = 768
        self.c = c

        # Positional encoding
        positional_encoding = gen_pos_encoding(int(input_length / self.c + 1), num_input_channels * self.c)
        self.register_buffer('positional_encoding', positional_encoding)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_input_channels * self.c, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_input_channels * self.c))

        self.proj_layer = nn.Linear(num_input_channels * self.c, num_output_channels)

    def forward(self, x):
        """
        Parameters:
        - x (torch.Tensor): A (batch_size, num_input_channels, input_length) tensor containing the input sEEG data.
        Returns:
        - x (torch.Tensor): A (batch_size, num_output_channels) tensor containing the sEEG embedding.
        """
        # Reshape the input to (batch_size, input_length / c, num_input_channels * c)
        x = x.view(x.shape[0], x.shape[1], -1, self.c)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x[:, 0, :]

        x = self.proj_layer(x)
        return x


# class SEEGEncoderChaFirst(SEEGEncoder):
#     """
#     A Transformer Encoder for sEEG data with channels matching first.
#     Parameters:
#     - num_input_channels (int): The number of input channels in the sEEG data.
#     - num_output_channels (int): The number of output channels.
#     - input_length (int): The length of the padded input sequence.
#     - output_length (int): The length of the output sequence.
#     - num_heads (int): The number of heads in the multi-head attention.
#     - num_encoder_layers (int): The number of encoder layers in the transformer.
#     - dim_feedforward (int): The dimension of the feedforward network in the transformer.
#     """
#     def __init__(self, num_heads=2, num_encoder_layers=6, dim_feedforward=2048):
#         super().__init__()
#
#         # Positional encoding
#         positional_encoding = gen_pos_encoding(self.input_length, self.num_output_channels)
#         self.register_buffer('positional_encoding', positional_encoding)
#
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_output_channels, nhead=num_heads,
#                                                    dim_feedforward=dim_feedforward, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#
#         self.layer_norm_1 = nn.LayerNorm(self.input_length)
#         self.layer_norm_2 = nn.LayerNorm(self.output_length)
#         self.layer_norm_3 = nn.LayerNorm(self.input_length)
#
#     def forward(self, x):
#         """
#         Parameters:
#         - x (torch.Tensor): A (batch_size, num_input_channels, input_length) tensor containing the input sEEG data.
#         Returns:
#         - x (torch.Tensor): A (batch_size, output_length, num_output_channels) tensor containing the output
#         sequence.
#         """
#         x = x.permute(0, 2, 1)
#         x = self.channel_matching_layer(x)
#         x = x.permute(0, 2, 1)
#         x = self.layer_norm_3(x)
#         x = self.relu(x)
#         x = x.permute(0, 2, 1)
#         x = x + self.positional_encoding
#         x = self.transformer_encoder(x)
#         x = x.permute(0, 2, 1)
#         x = self.layer_norm_1(x)
#         x = self.relu(x)
#         x = self.length_matching_layer(x)
#         x = self.layer_norm_2(x)
#         x = x.permute(0, 2, 1)
#         return x
#
#
# class SEEGEncoderLenChaFirst(SEEGEncoder):
#     """
#     A Transformer Encoder for sEEG data with length and channels matching first.
#
#     Parameters:
#     - num_input_channels (int): The number of input channels in the sEEG data.
#     - num_output_channels (int): The number of output channels.
#     - input_length (int): The length of the padded input sequence.
#     - output_length (int): The length of the output sequence.
#     - num_heads (int): The number of heads in the multi-head attention.
#     - num_encoder_layers (int): The number of encoder layers in the transformer.
#     - dim_feedforward (int): The dimension of the feedforward network in the transformer.
#     """
#
#     def __init__(self, num_heads=2, num_encoder_layers=6, dim_feedforward=2048):
#         super().__init__()
#
#         # Positional encoding
#         positional_encoding = gen_pos_encoding(self.output_length, self.num_output_channels)
#         self.register_buffer('positional_encoding', positional_encoding)
#
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_output_channels, nhead=num_heads,
#                                                    dim_feedforward=dim_feedforward, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#
#         self.layer_norm_1 = nn.LayerNorm(self.output_length)
#         self.layer_norm_2 = nn.LayerNorm(self.output_length)
#         self.layer_norm_3 = nn.LayerNorm(self.output_length)
#
#     def forward(self, x):
#         """
#         Parameters:
#         - x (torch.Tensor): A (batch_size, num_input_channels, input_length) tensor containing the input sEEG data.
#
#         Returns:
#         - x (torch.Tensor): A (batch_size, output_length, num_output_channels) tensor containing the output
#         sequence.
#         """
#         x = self.length_matching_layer(x)
#         x = self.layer_norm_1(x)
#         x = self.relu(x)
#         x = x.permute(0, 2, 1)
#         x = self.channel_matching_layer(x)
#         x = x.permute(0, 2, 1)
#         x = self.layer_norm_2(x)
#         x = x.permute(0, 2, 1)
#         x = self.relu(x)
#         x = x + self.positional_encoding
#         x = self.transformer_encoder(x)
#         x = x.permute(0, 2, 1)
#         x = self.layer_norm_3(x)
#         x = x.permute(0, 2, 1)
#         return x


if __name__ == '__main__':
    import torch
    from util.experiment import print_num_params

    num_heads = 6
    num_encoder_layers = 6
    dim_feedforward = 2048
    num_input_channels = 84
    input_length = 5120

    model = SEEGEncoder(num_heads, num_encoder_layers, dim_feedforward, num_input_channels, input_length)
    print_num_params(model)

    seegs = torch.randn(2, 84, 5120)

    with torch.no_grad():
        output = model(seegs)
    assert output.shape == (2, 768)

    model = SEEGEncoderProj(num_heads, num_encoder_layers, dim_feedforward, num_input_channels, input_length)
    print_num_params(model)
    with torch.no_grad():
        output = model(seegs)
    assert output.shape == (2, 768)
