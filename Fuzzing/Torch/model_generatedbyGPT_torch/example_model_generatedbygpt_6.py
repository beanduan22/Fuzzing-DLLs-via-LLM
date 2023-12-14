import torch
import torch.nn as nn
import torch.nn.functional as F

class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.instance_norm2d = nn.InstanceNorm2d(3)

        self.used_apis = [
            'torch.nn.InstanceNorm2d',
            'torch.Tensor.round_',
            'torch.Tensor.cos',
            'torch.Tensor.lt_',
            'torch.Tensor.trunc_',
            'torch.Tensor.arctan',
            'torch.Tensor.masked_fill',
            'torch.fmax',
            'torch.fft.ifft2',
            'torch.mm'
        ]

    def forward(self, x):
        results = {}
        # Normalization
        x = self.instance_norm2d(x)

        x = x.round_()

        x = torch.cos(x)

        x = x.lt_(0.5)

        x = x.trunc_()

        x = torch.arctan(x)

        x = x.masked_fill(x < 0.5, 0)

        x = torch.fmax(x, torch.tensor(0.5))

        x = x.mean(dim=(0, 1))

        x = torch.mm(x, x, out=x)

        output = torch.fft.ifft2(x)

        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted
