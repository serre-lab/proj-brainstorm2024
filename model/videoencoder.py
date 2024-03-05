import torch
import torch.nn as nn
from transformers import VideoMAEModel
from util.data import process_video


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(150, 1)

    def forward(self, x):
        x = self.linear(x.permute(0, 2, 1))
        return x