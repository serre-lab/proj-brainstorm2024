import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(234 * 5120, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 234 * 5120),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_channels = x.size(1)

        x = x.view(batch_size, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batch_size, num_channels, -1)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoEncoder().to(device)

    input = torch.randn(90, 234, 5120).to(device)

    with torch.no_grad():
        output = model(input)

    assert input.size() == output.size()
