import torch
import torch.nn as nn
from transformers import VideoMAEModel
from util.data import process_video


class VideoEncoder(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(ckpt)
        self.freeze_parameters()
        self.compress_layer = nn.Sequential(
            nn.Conv1d(1568, 128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1, stride=1),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.model(x).last_hidden_state
        x = self.compress_layer(x)
        x = x.squeeze(1)
        return x

    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    import glob

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = "MCG-NJU/videomae-base"
    model = VideoEncoder(ckpt).to(device)
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
