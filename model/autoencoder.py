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

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial 1D convolutions on the temporal dimension
        self.initial_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # Can add more layers or adjust as needed
        )

        # Encoder 2D convolutions for spatial-temporal features
        self.encoder_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(True),
            # Additional layers can be added
        )

        # Decoder 2D convolutions
        self.decoder_conv2d = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ReLU(True),
            # Adjust or add layers as needed
        )

        # Final 1D deconvolutions to restore original temporal dimension
        self.final_deconv1d = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # Adjust output_padding as necessary
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        # x shape expected to be (batch, 234, 1, 5120)
        batch_size, electrodes, channels, length = x.size()

        x = x.view(-1, channels, length)  # (batch*234, 1, 5120)
        x = self.initial_conv1d(x)  # Apply initial 1D convolutions

        x = x.view(batch_size, electrodes, -1, x.size(-1))  # Reshape to (batch, 234, channels, new_length)
        x = x.permute(0, 2, 1, 3)  # Rearrange to (batch, channels, electrodes, new_length) for 2D conv
        embed = self.encoder_conv2d(x)  # Apply 2D convolutions

        x = self.decoder_conv2d(embed)
        x = x.permute(0, 2, 1, 3)  # Rearrange back to (batch, electrodes, channels, new_length)
        x = x.reshape(batch_size * electrodes, x.size(2), x.size(-1))  # Flatten batch and electrode dimensions

        x = self.final_deconv1d(x)
        x = x.view(batch_size, electrodes, channels, -1)

        return x.squeeze(), embed


if __name__ == '__main__':
    from util.loss import recon_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = AutoEncoder().to(device)
    model = ConvAutoEncoder().to(device)

    input = torch.randn(90, 234, 5120).to(device)

    with torch.no_grad():
        output, embed = model(input)

    assert input.size() == output.size()
    # assert embed.size() == (90, 32)

    loss = recon_loss(input, output)
    print(f'Reconstruction loss: {loss.item()}')
