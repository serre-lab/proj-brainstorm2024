import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128 * 14 * 20, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 30)


    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = nn.functional.relu(self.linear4(x))
        return self.linear5(x)


if __name__ == '__main__':
    import torch
    classifier = LinearClassifier()
    input = torch.randn(32, 128, 14, 20)
    output = classifier(input)
    assert output.size() == (32, 30)
