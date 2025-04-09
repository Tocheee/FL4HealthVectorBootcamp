import torch

# class MLP(torch.nn.Module):
#     def __init__(self, input_dim: int, output_dim: int) -> None:
#         super().__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         outputs = torch.sigmoid(self.linear(x))
#         return outputs

class MLPNet(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims=[128, 64], output_dim=1):
        super(MLPNet, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            layers.append(torch.nn.BatchNorm1d(dims[i + 1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.3))
        layers.append(torch.nn.Linear(dims[-1], output_dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(1)  # For BCEWithLogitsLoss

