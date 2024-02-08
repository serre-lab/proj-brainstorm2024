import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128 * 14 * 20, 30)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.linear(x)


if __name__ == '__main__':
    import torch
    classifier = LinearClassifier()
    input = torch.randn(32, 128, 14, 20)
    output = classifier(input)
    assert output.size() == (32, 30)
