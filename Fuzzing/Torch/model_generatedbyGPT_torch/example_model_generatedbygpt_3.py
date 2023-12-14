import torch
import torch.nn as nn
import torch.nn.functional as F


class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initializing necessary layers
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=3)
        self.unfold = nn.Unfold(kernel_size=3)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Manually selected APIs for usage

        self.used_apis = [
            'torch.Tensor.fill_',
            'torch.Tensor.frac',
            'torch.Tensor.arccos_',
            'torch.Tensor.heaviside',
            'torch.std_mean',
            'torch.nn.Unfold',
            'torch.nn.GroupNorm',
            'torch.nn.PixelShuffle',
            'toech.nn.LogSoftmax',
            'torch.package.PackageExporter'
        ]

    def forward(self, x):
        x = self.group_norm(x)

        x = self.unfold(x)

        x = self.pixel_shuffle(x)

        x = self.log_softmax(x)

        x.fill_(1.0)

        x_frac = x.frac()

        x_arccos = torch.clamp(x_frac, -1, 1).arccos_()

        x_heaviside = x_arccos.heaviside(x_arccos)

        output = torch.std_mean(x_heaviside)

        torch.package.PackageExporter(output, './test.zip')

        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted