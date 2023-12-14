import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.distributions.transforms as transforms
import torch.optim


class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool1d = nn.AvgPool1d(kernel_size=2)
        self.avg_pool3d = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.channel_shuffle = nn.ChannelShuffle(groups=3)

        self.used_apis = [
            'torch.Tensor.diag',
            'torch.Tensor.div',
            'torch.Tensor.floor_divide_',
            'torch.conj',
            'torch.log2',
            'torch.enable_grad',
            'torch.not_equal',
            'torch.nn.ChannelShuffle',
            'torch.nn.functional.avg_pool1d',
            'torch.nn.functional.avg_pool3d',
            'torch.fft.rfft2',
            'torch.fft.irfft2'
        ]

    def forward(self, x):
        device = x.device

        x = x.view(-1, 3, 32 * 32)

        x = self.avg_pool1d(x).to(device)

        x = self.channel_shuffle(x)

        # Reshape for AvgPool3d
        x = x.view(-1, 3, 8, 4, 4)
        x = self.avg_pool3d(x).to(device)

        # Apply various operations
        x = torch.diag(x.view(-1)).to(device)

        x = torch.div(x, 2).to(device)

        x.floor_divide_(2)

        x = torch.conj(x).to(device)

        x = torch.log2(x).to(device)

        # FFT operations
        x = torch.fft.rfft2(x).to(device)

        x_irfft2 = torch.fft.irfft2(x).to(device)

        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return x_irfft2, used_apis_sorted

