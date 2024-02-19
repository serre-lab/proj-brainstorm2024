import torch.nn as nn
from model.autoencoder import ConvAutoEncoder


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear1 = nn.Linear(128 * 14 * 20, 30)
        self.linear1 = nn.Linear(1433600, 30)
        # self.linear1 = nn.Linear(512 * 20 * 20, 30)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.linear1(x)


class GAPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, 30)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class E2eClassifier(nn.Module):
    def __init__(self, num_electrodes):
        super().__init__()
        self.autoencoder = ConvAutoEncoder(num_electrodes)
        # self.classifier = LinearClassifier()
        self.classifier = GAPClassifier()

    def forward(self, x, id):
        recon, embed = self.autoencoder(x, id)
        return self.classifier(embed), recon


if __name__ == '__main__':
    import torch
    # classifier = LinearClassifier()
    # input = torch.randn(32, 128, 14, 20)
    # output = classifier(input)
    # assert output.size() == (32, 30)

    from util.data import ID_2_IDX_CHANNEL
    num_electrodes = [ID_2_IDX_CHANNEL[key][1] for key in
                      sorted(ID_2_IDX_CHANNEL, key=lambda x: ID_2_IDX_CHANNEL[x][0])]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = E2eClassifier(num_electrodes).to(device)
    input = torch.randn(2, 234, 5120).to(device)
    with torch.no_grad():
        logits, recon = classifier(input, 1)
    assert logits.size() == (2, 30)
    # assert recon.size() == (90, 234, 5120)
