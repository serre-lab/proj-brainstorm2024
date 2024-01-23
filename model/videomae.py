import av
import torch
import torch.nn as nn
from transformers import VideoMAEModel
from transformers import VideoMAEImageProcessor
from util.data import get_frames, sample_frame_indices


class VideoMAE(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(ckpt)

    def forward(self, **kwargs):
        return self.model(**kwargs).last_hidden_state


if __name__ == '__main__':
    ckpt = "MCG-NJU/videomae-base"
    model = VideoMAE(ckpt)

    file_path = '../data/dev/eating_spaghetti.mp4'
    container = av.open(file_path)

    # sample 16 frames
    indices = sample_frame_indices(num_frame_2_sample=16, frame_sample_rate=2,
                                   max_end_frame_idx=container.streams.video[0].frames)
    frames = get_frames(container, indices)

    image_processor = VideoMAEImageProcessor.from_pretrained(ckpt)

    inputs = image_processor(list(frames), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    print(outputs.shape)
