import torch


class MLPNet(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),

            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
