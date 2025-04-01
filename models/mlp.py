import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, output_dim)
        self.activation = nn.Softplus()

    def forward(self, input):
        input = input.view(input.shape[0], -1)
        y_hat = self.activation(self.fc1(input))
        y_hat = self.fc2(y_hat)
        return y_hat
