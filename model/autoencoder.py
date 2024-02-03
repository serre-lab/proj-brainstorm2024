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
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_channels = x.size(1)

        x = x.view(batch_size, -1)
        embedding = self.encoder(x)
        output = self.decoder(embedding)
        output = output.view(batch_size, num_channels, -1)
        return output, embedding


if __name__ == '__main__':
    from util.loss import recon_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoEncoder().to(device)

    input = torch.randn(90, 234, 5120).to(device)

    with torch.no_grad():
        output, embed = model(input)

    assert input.size() == output.size()
    assert embed.size() == (90, 32)

    loss = recon_loss(input, output)
    print(f'Reconstruction loss: {loss.item()}')
