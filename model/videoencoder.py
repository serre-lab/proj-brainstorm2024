import torch
import torch.nn as nn
from util.model import gen_pos_encoding


class VideoEncoderVdFt(nn.Module):
    """
    Video Encoder for the finetuned VideoMAE features
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 768)

    def forward(self, x):
        x = self.proj(x)
        return x


class VideoEncoderDino(nn.Module):
    """
    Video Encoder for the DINO features
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.proj = nn.Linear(768, 768)

    def forward(self, x):
        x = self.linear(x.permute(0, 2, 1))
        x = x.squeeze(-1)
        x = self.proj(x)
        return x


class VideoEncoderDinoScene(nn.Module):
    """
    Video Encoder for the DINO features that handles variable sequence length
    """
    def __init__(self):
        super().__init__()
        max_length = 201

        positional_encoding = gen_pos_encoding(max_length + 1, 768)
        self.register_buffer('positional_encoding', positional_encoding)

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=6, dim_feedforward=1024, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))

        self.proj = nn.Linear(768, 768)

    def forward(self, x, padding_mask):
        """
        Parameters:
        - x (torch.Tensor): A (batch_size, max_length, 768) tensor containing the input DINO features.
        - padding_mask (torch.Tensor): A (batch_size, max_length) boolean tensor containing the mask for the padding.
        True indicates a padding position and False indicates a valid data position.
        Returns:
        - x (torch.Tensor): A (batch_size, 768) tensor containing the DINO embedding.
        """
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        padding_mask = torch.cat((torch.zeros(x.shape[0], 1, dtype=torch.bool).to(x.device), padding_mask), dim=1)

        x += self.positional_encoding
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = x[:, 0, :]

        x = self.proj(x)
        return x


# class VideoEncoderProj(nn.Module):
#     def __init__(self, ckpt, num_heads=6, num_encoder_layers=6, dim_feedforward=2048):
#         super().__init__()
#         self.model = VideoMAEModel.from_pretrained(ckpt)
#         self.freeze_parameters()
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
#
#         encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#
#         self.proj_layer = nn.Linear(768, 768)
#
#     def forward(self, x):
#         # Get the pretrained features
#         x = self.model(x).last_hidden_state
#
#         # Aggregate the features
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = self.transformer_encoder(x)
#         x = x[:, 0, :]
#
#         # Project the features to the multimodal space
#         x = self.proj_layer(x)
#         return x
#
#     def freeze_parameters(self):
#         for param in self.model.parameters():
#             param.requires_grad = False


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = VideoEncoderDino().to(device)
    # inputs = torch.rand(10, 150, 768).to(device)
    # with torch.no_grad():
    #     outputs = model(inputs)
    # assert outputs.shape == (10, 768)
    #
    # model = VideoEncoderVdFt().to(device)
    # inputs = torch.rand(10, 768).to(device)
    # with torch.no_grad():
    #     outputs = model(inputs)
    # assert outputs.shape == (10, 768)

    model = VideoEncoderDinoScene().to(device)
    inputs = torch.rand(10, 201, 768).to(device)
    padding_mask = torch.zeros(10, 201).to(device)
    padding_mask[:, 100:] = 1
    with torch.no_grad():
        outputs = model(inputs, padding_mask)
    assert outputs.shape == (10, 768)
