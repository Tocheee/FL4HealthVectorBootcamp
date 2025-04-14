import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.sigmoid(self.linear(x))
        return outputs



class BinaryModel(nn.Module):
    def __init__(self, input_dim: int):
        super(BinaryModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)

        self.act = nn.GELU()  # Better for deeper networks
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.act(self.bn1(self.fc1(x))))
        x = self.dropout(self.act(self.bn2(self.fc2(x))))
        x = self.dropout(self.act(self.bn3(self.fc3(x))))
        x = self.out(x).reshape(-1)
        return x
