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


class ConvAutoEncoder(nn.Module):
    def __init__(self, num_electrodes):
        super().__init__()
        self.fc1 = nn.ModuleList([nn.Linear(num_electrodes[i], 160) for i in range(len(num_electrodes))])
        self.fc2 = nn.ModuleList([nn.Linear(160, num_electrodes[i]) for i in range(len(num_electrodes))])

        # Initial 1D convolutions on the temporal dimension with MaxPooling
        self.initial_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            # nn.Dropout(p=0.001),  # Add dropout here, adjust p as needed
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            # nn.Dropout(p=0.001),  # Add dropout here, adjust p as needed
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            # nn.Dropout(p=.001),  # Add dropout here, adjust p as needed

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            # nn.Dropout(p=.001),  # Add dropout here, adjust p as needed

            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # nn.Dropout(p=.001),  # Add dropout here, adjust p as needed

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # nn.Dropout(p=.001),  # Add dropout here, adjust p as needed

            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout(p=.001),  # Add dropout here, adjust p as needed

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout(p=.001),  # Add dropout here, adjust p as needed

            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout(p=.001),  # Add dropout here, adjust p as needed

            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout(p=.001),  # Add dropout here, adjust p as needed,

            nn.MaxPool1d(kernel_size=2, stride=2),
        )


        # Encoder 2D convolutions for spatial-temporal features with MaxPooling
        self.encoder_conv2d = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            # nn.Dropout2d(p=.001),  # Adjust p as needed
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            # nn.Dropout2d(p=.001),  # Adjust p as needed
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            # nn.Dropout2d(p=.001),  # Adjust p as needed
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            # nn.Dropout2d(p=.001),  # Adjust p as needed
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(True),
            # nn.Dropout2d(p=.001),  # Adjust p as needed
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512), 
            nn.ReLU(True),
            # nn.Dropout2d(p=.001),  # Adjust p as needed
            nn.MaxPool2d(kernel_size=(2, 2)),
        )


        # Decoder 2D convolutions with Upsampling followed by Convolution
        self.decoder_conv2d = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(256), 
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(128), 
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
        )

        # Final 1D deconvolutions to restore original temporal dimension with Upsampling followed by Convolution
        self.final_deconv1d = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(True),

            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(True),
        )


    def forward(self, x, id):
        x = x.unsqueeze(2)  # Expecting shape to be (batch, 234, 1, 5120)
        batch_size, electrodes, channels, length = x.size()

        x = x.view(-1, channels, length)  # Flatten to (batch*234, 1, 5120)
        x = self.initial_conv1d(x)  # Apply initial 1D convolutions with MaxPooling

        # Reshape and permute for 2D convolutions
        x = x.view(batch_size, electrodes, -1, x.size(-1))
        x = x.permute(0, 2, 3, 1)
        x = self.fc1[id](x)
        x = x.permute(0, 1, 3, 2)
        embed = self.encoder_conv2d(x)  # Apply 2D convolutions with MaxPooling

        x = self.decoder_conv2d(embed)
        x = x.permute(0, 1, 3, 2)
        x = self.fc2[id](x)
        x = x.permute(0, 3, 1, 2)  # Rearrange for decoding
        x = x.reshape(batch_size * electrodes, x.size(2), x.size(-1))  # Flatten for 1D deconvolutions

        x = self.final_deconv1d(x)
        x = x.view(batch_size, electrodes, channels, -1)

        return x.squeeze(), embed


if __name__ == '__main__':
    from util.loss import recon_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = AutoEncoder().to(device)
    model = ConvAutoEncoder(num_electrodes=[234, 233]).to(device)

    input = torch.randn(5, 234, 5120).to(device)

    with torch.no_grad():
        output, embed = model(input, 0)

    assert input.size() == output.size()
    # assert embed.size() == (90, 32)

    loss = recon_loss(input, output)
    print(f'Reconstruction loss: {loss.item()}')
