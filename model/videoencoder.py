import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEForVideoClassification
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional, Tuple, Union
from util.data import process_video


class VideoEncoderCls(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.model = VideoMAECls.from_pretrained(ckpt)
        self.freeze_parameters()
        self.linear = nn.Linear(768, 768)

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x

    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False


class VideoEncoder(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(ckpt)
        self.linear = nn.Linear(1568, 1)

    def forward(self, x):
        x = self.model(x).last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = x.permute(0, 2, 1)
        return x

    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False


class VideoEncoderProj(nn.Module):
    def __init__(self, ckpt, num_heads=6, num_encoder_layers=6, dim_feedforward=2048):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(ckpt)
        self.freeze_parameters()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.proj_layer = nn.Linear(768, 768)

    def forward(self, x):
        # Get the pretrained features
        x = self.model(x).last_hidden_state

        # Aggregate the features
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]

        # Project the features to the multimodal space
        x = self.proj_layer(x)
        return x

    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False


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


if __name__ == '__main__':
    import glob

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #ckpt = "MCG-NJU/videomae-base"
    ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
    model = VideoEncoderCls(ckpt).to(device)
    from util.experiment import print_num_params
    print_num_params(model)

    # Process the videos
    num_frame_2_sample = 16
    video_paths = glob.glob('../data/dev/Movie Clips/*.avi')[:5]
    inputs = [process_video(video_path, ckpt, num_frame_2_sample) for video_path in video_paths]
    batched_inputs = torch.stack(inputs, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batched_inputs)

    assert outputs.shape == (len(video_paths), 768)

    # assert outputs.shape == (len(video_paths), 1568, 768)

    # model = VideoEncoderProj(ckpt).to(device)
    # print_num_params(model)
    # with torch.no_grad():
    #     outputs = model(batched_inputs)
    #
    # assert outputs.shape == (len(video_paths), 768)
