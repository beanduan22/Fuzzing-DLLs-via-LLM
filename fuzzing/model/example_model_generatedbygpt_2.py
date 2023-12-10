import torch
import torch.nn as nn
import torch.nn.functional as F

class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.avgpool2d = nn.AvgPool2d(kernel_size=2)

        self.used_apis = [
            'torch.nn.Conv2d',
            'torch.nn.AvgPool2d',
            'torch.Tensor.new_ones',
            'torch.Tensor.log2',
            'torch.Tensor.nanmean',
            'torch.Tensor.logaddexp',
            'torch.Tensor.scatter',
            'torch.tensor',
            'torch.Tensor.view',
            'torch.Tensor.size',
            'torch.Tensor.unsqueeze',
            'torch.Tensor.squeeze',
            'torch.amax'
        ]

    def forward(self, x):
        device = x.device

        x = self.conv2d(x)

        x = self.avgpool2d(x)

        x_flattened = x.view(x.size(0), -1)

        x_new_ones = x_flattened.new_ones(x_flattened.shape)

        x_log2 = torch.log2(x_new_ones)

        x_nanmean = torch.nanmean(x_flattened, dim=1)

        x_amax = torch.amax(x_flattened, dim=1)

        combined_result = x_nanmean + x_log2.mean(dim=1) + x_amax

        unsqueeze_result = combined_result.unsqueeze(1)

        squeezed_result = unsqueeze_result.squeeze()

        index = torch.tensor([1, 1]).to(device)

        output = torch.Tensor.scatter(squeezed_result, 0, index, squeezed_result)

        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted
