import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

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


# class ConvAutoEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # Initial 1D convolutions on the temporal dimension with MaxPooling
#         self.initial_conv1d = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=4, stride=4),  # Adding MaxPool1d
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=4, stride=4),  # Adding MaxPool1d
#         )
#
#         # Encoder 2D convolutions for spatial-temporal features with MaxPooling
#         self.encoder_conv2d = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=(4, 4)),  # Adding MaxPool2d
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=(4, 4)),  # Adding MaxPool2d
#         )
#
#         # Decoder 2D convolutions with Upsampling followed by Convolution
#         self.decoder_conv2d = nn.Sequential(
#             nn.Upsample(scale_factor=(4, 4)),  # Upsampling
#             nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=(2, 1)),
#             nn.ReLU(True),
#             nn.Upsample(scale_factor=(4, 4)),  # Upsampling
#             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=(2, 1)),
#             nn.ReLU(True),
#
#         )
#
#         # Final 1D deconvolutions to restore original temporal dimension with Upsampling followed by Convolution
#         self.final_deconv1d = nn.Sequential(
#             nn.Upsample(scale_factor=4),  # Upsampling
#             nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(True),
#             nn.Upsample(scale_factor=4),  # Upsampling
#             nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
#             nn.ReLU(True),
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(2)  # Expecting shape to be (batch, 234, 1, 5120)
#         batch_size, electrodes, channels, length = x.size()
#
#         x = x.view(-1, channels, length)  # Flatten to (batch*234, 1, 5120)
#         x = self.initial_conv1d(x)  # Apply initial 1D convolutions with MaxPooling
#
#         # Reshape and permute for 2D convolutions
#         x = x.view(batch_size, electrodes, -1, x.size(-1))
#         x = x.permute(0, 2, 1, 3)
#         embed = self.encoder_conv2d(x)  # Apply 2D convolutions with MaxPooling
#
#         x = self.decoder_conv2d(embed)
#         x = x.permute(0, 2, 1, 3)  # Rearrange for decoding
#         x = x.reshape(batch_size * electrodes, x.size(2), x.size(-1))  # Flatten for 1D deconvolutions
#
#         x = self.final_deconv1d(x)
#         x = x.view(batch_size, electrodes, channels, -1)
#
#         return x.squeeze(), embed

class DownConv1d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, p=0.001):
        super(DownConv1d_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x

class DownConv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, p=0.001):
        super(DownConv2d_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class UpConv1d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, p=0.001):
        super(UpConv1d_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class UpConv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, p=0.001):
        super(UpConv2d_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class ConvAutoEncoder(nn.Module):
    def __init__(self, num_electrodes):
        super().__init__()
        self.fc1 = nn.ModuleList([nn.Linear(num_electrodes[i], 160) for i in range(len(num_electrodes))])
        self.fc2 = nn.ModuleList([nn.Linear(160, num_electrodes[i]) for i in range(len(num_electrodes))])

        # Initial 1D convolutions on the temporal dimension with MaxPooling
        self.Down_1D_1 = DownConv1d_block(1, 8)
        self.Down_1D_2 = DownConv1d_block(8, 16)
        self.Down_1D_3 = DownConv1d_block(16, 32)
        self.Down_1D_4 = DownConv1d_block(32, 64)
        self.Down_1D_5 = DownConv1d_block(64, 64)

        self.Down_2D_1 = DownConv2d_block(64, 128)
        self.Down_2D_2 = DownConv2d_block(128, 256)
        self.Down_2D_3 = DownConv2d_block(256, 512)

        self.Up_2D_1 = UpConv2d_block(512, 256)  # Note the channel adjustments for concatenation
        self.Up_2D_2 = UpConv2d_block(256 + 256, 128)
        self.Up_2D_3 = UpConv2d_block(128 + 128, 64)  # Transition back to 1D happens after this

        self.Up_1D_1 = UpConv1d_block(64 + 64, 64)
        self.Up_1D_2 = UpConv1d_block(64, 32)
        self.Up_1D_3 = UpConv1d_block(32, 16)
        self.Up_1D_4 = UpConv1d_block(16, 8)
        self.Up_1D_5 = UpConv1d_block(8, 1)

    def forward(self, x, id):
        x = x.unsqueeze(2)  # Expecting shape to be (batch, 234, 1, 5120)
        batch_size, electrodes, channels, length = x.size()

        x = x.view(-1, channels, length)  # Flatten to (batch*234, 1, 5120)
        x = self.Down_1D_1(x)
        x = self.Down_1D_2(x)
        x = self.Down_1D_3(x)
        x = self.Down_1D_4(x)
        x3 = self.Down_1D_5(x)

        # Reshape and permute for 2D convolutions
        x = x3.view(batch_size, electrodes, -1, x3.size(-1))
        x = x.permute(0, 2, 3, 1)
        x = self.fc1[id](x)
        x = x.permute(0, 1, 3, 2)
        x4 = self.Down_2D_1(x)
        x5 = self.Down_2D_2(x4)
        embed = self.Down_2D_3(x5)  # Apply 2D convolutions with MaxPooling

        x = self.Up_2D_1(embed)
        x = self.Up_2D_2(torch.cat([x, x5], dim=1))
        x = self.Up_2D_3(torch.cat([x, x4], dim=1))

        x = x.permute(0, 1, 3, 2)
        x = self.fc2[id](x)
        x = x.permute(0, 3, 1, 2)  # Rearrange for decoding
        x = x.reshape(batch_size * electrodes, x.size(2), x.size(-1))  # Flatten for 1D deconvolutions

        x = self.Up_1D_1(torch.cat([x, x3], dim=1))
        x = self.Up_1D_2(x)
        x = self.Up_1D_3(x)
        x = self.Up_1D_4(x)
        x = self.Up_1D_5(x)
        x = x.view(batch_size, electrodes, channels, -1)

        return x.squeeze(), torch.cat([x3.view(batch_size, electrodes, -1, x3.size(-1)).flatten(start_dim=1), x4.flatten(start_dim=1), x5.flatten(start_dim=1), embed.flatten(start_dim=1)], dim=1)


if __name__ == '__main__':
    from util.loss import recon_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = AutoEncoder().to(device)
    model = ConvAutoEncoder(num_electrodes=[234, 233]).to(device)
    # print number of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    input = torch.randn(5, 234, 5120).to(device)

    with torch.no_grad():
        output, embed = model(input, 0)

    assert input.size() == output.size()
    # assert embed.size() == (90, 32)

    loss = recon_loss(input, output)
    print(f'Reconstruction loss: {loss.item()}')
