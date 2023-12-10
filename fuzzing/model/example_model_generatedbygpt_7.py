import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.transforms as T

class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32 * 32 * 3, 128)
        self.pool = nn.MaxPool1d(2, stride=2)

        # APIs used in the forward method
        self.used_apis = [
            'torch.Tensor.not_equal_',
            'torch.Tensor.conj_physical_',
            'torch.sin',
            'torch.Tensor.repeat_interleave',
            'torch.Tensor.asin',
            'torch.arcsin',
            'torch.arccos',
            'torch.outer',
            'torch.remainder',
            'torch.heaviside',
            'torch.nn.functional.relu_'
        ]

    def forward(self, x):
        # Reshape and apply linear layer
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        x = F.relu_(x)

        x = x.repeat_interleave(2, dim=1)

        x = torch.asin(x)

        x = x.not_equal_(0.5)

        x = torch.sin(x)

        x = torch.arcsin(x)

        x = torch.arccos(x)

        x = x.conj_physical_()

        x = self.pool(x)

        x = torch.outer(x[:, 0], x[:, 1])

        x = torch.remainder(x, 2)

        x = torch.heaviside(x, x)

        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)
        return x, used_apis_sorted