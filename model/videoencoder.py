import torch
import torch.nn as nn
from transformers import VideoMAEModel
from util.data import process_video


class VideoEncoder(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(ckpt)

    def forward(self, **kwargs):
        return self.model(**kwargs).last_hidden_state


if __name__ == '__main__':
    ckpt = "MCG-NJU/videomae-base"
    model = VideoEncoder(ckpt)

    # Process the videos
    num_frame_2_sample = 16
    frame_sample_rate = 2
    video_paths = ['../data/dev/eating_spaghetti.mp4', '../data/dev/eating_spaghetti-copy.mp4']
    inputs = [process_video(video_path, ckpt, num_frame_2_sample, frame_sample_rate) for video_path in video_paths]
    batched_inputs = {key: torch.cat([input[key] for input in inputs], dim=0) for key in inputs[0]}

    with torch.no_grad():
        outputs = model(**batched_inputs)

    print(outputs.shape)
