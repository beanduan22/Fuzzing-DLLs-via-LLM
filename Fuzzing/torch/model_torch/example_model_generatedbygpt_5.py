import torch
import torch.nn as nn
import torch.nn.functional as F

class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32 * 32 * 3, 10)

        # List only APIs used in the forward method
        self.used_apis = [
            'torch.Tensor.atan',
            'torch.Tensor.t',
            'torch.Tensor.ceil_',
            'torch.Tensor.msort',
            'torch.Tensor.erf_',
            'torch.Tensor.arcsinh',
            'torch.Tensor.neg_',
            'torch.nn.functional.celu',
            'torch.nn.Linear',
            'torch.corrcoef'
        ]

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = self.linear(x)

        x.atan_()

        x.ceil_()

        x = torch.fmax(x, torch.tensor(0.1))

        x = torch.msort(x)

        x.erf_()

        x.arcsinh_()

        x.neg_()

        x = F.celu(x)

        x = x.t()

        output = torch.corrcoef(x)

        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)
        return output, used_apis_sorted