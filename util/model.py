import torch
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional, Tuple, Union
from transformers import VideoMAEForVideoClassification


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


class VideoMAECls(VideoMAEForVideoClassification):
    """
    VideoMAEForVideoClassification with a custom forward method to output the CLS token.
    """
    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        ```"""
        outputs = self.videomae(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = sequence_output[:, 0]

        return sequence_output


# class CustomMultiheadAttention(nn.Module):
#     def __init__(self, dim, q_k_dim, v_dim, num_heads):
#         super().__init__()
#         self.q_k_dim = q_k_dim
#         self.v_dim = v_dim
#         self.num_heads = num_heads
#
#         self.qk_proj = nn.Linear(dim, q_k_dim * 2)
#         self.v_proj = nn.Linear(dim, v_dim)
#         self.out_proj = nn.Linear(v_dim, v_dim)
#
#         self.scale = torch.sqrt(torch.FloatTensor([q_k_dim // num_heads]))
#
#     def forward(self, x):
#         B, N, C = x.shape
#
#         qk = self.qk_proj(x)
#         qk = qk.reshape(B, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k = qk[0], qk[1]
#         v = self.v_proj(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) / self.scale
#         attn = attn.softmax(dim=-1)
#
#         out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#         out = self.out_proj(out)
#         return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = "sayakpaul/videomae-base-finetuned-kinetics-finetuned-ucf101-subset"
    model = VideoMAECls.from_pretrained(ckpt).to(device)

    # Process the videos
    inputs = torch.rand(5, 16, 3, 224, 224).to(device)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (5, 768)
