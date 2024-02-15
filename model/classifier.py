import torch.nn as nn
from model.autoencoder import ConvAutoEncoder


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear1 = nn.Linear(128 * 14 * 20, 30)
        self.linear1 = nn.Linear(512 * 20 * 20, 30)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.linear1(x)


class E2eClassifier(nn.Module):
    def __init__(self, num_electrodes):
        super().__init__()
        self.autoencoder = ConvAutoEncoder(num_electrodes)
        self.classifier = LinearClassifier()

    def forward(self, x, id):
        recon, embed = self.autoencoder(x, id)
        return self.classifier(embed), recon


if __name__ == '__main__':
    import torch
    # classifier = LinearClassifier()
    # input = torch.randn(32, 128, 14, 20)
    # output = classifier(input)
    # assert output.size() == (32, 30)

    classifier = E2eClassifier()
    input = torch.randn(90, 234, 5120)
    with torch.no_grad():
        logits, recon = classifier(input)
    assert logits.size() == (90, 30)
    assert recon.size() == (90, 234, 5120)
