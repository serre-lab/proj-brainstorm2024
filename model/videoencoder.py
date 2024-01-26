import torch
import torch.nn as nn
from transformers import VideoMAEModel
from util.data import process_video


class VideoEncoder(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(ckpt)
        self.freeze_parameters()

    def forward(self, x):
        return self.model(x).last_hidden_state

    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    ckpt = "MCG-NJU/videomae-base"
    model = VideoEncoder(ckpt)

    # Process the videos
    num_frame_2_sample = 16
    frame_sample_rate = 2
    video_paths = ['../data/dev/eating_spaghetti.mp4', '../data/dev/eating_spaghetti-copy.mp4']
    inputs = [process_video(video_path, ckpt, num_frame_2_sample, frame_sample_rate) for video_path in video_paths]
    batched_inputs = torch.cat([input['pixel_values'] for input in inputs], dim=0)

    with torch.no_grad():
        outputs = model(batched_inputs)

    assert outputs.shape == (len(video_paths), 1568, 768)
