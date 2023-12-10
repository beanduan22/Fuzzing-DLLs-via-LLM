import torch
import torch.nn as nn
import torch.nn.functional as F

class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32 * 32 * 3, 100)
        self.frac_maxpool2d = nn.FractionalMaxPool2d(2, output_ratio=0.5)
        self.hardswish = nn.Hardswish()

        self.used_apis = [
            'torch.Tensor.share_memory_',
            'torch.Tensor.tanh_',
            'torch.hypot',
            'torch.negative',
            'torch.combinations',
            'torch.fmin',
            'torch.cos',
            'torch.nn.Linear',
            'torch.nn.FractionalMaxPool2d',
            'torch.nn.Hardswish',
            'torch.lobpcg'
        ]

    def forward(self, x):
        results = {}

        # Reshape and apply linear layer
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        # Apply FractionalMaxPool2d
        x = x.view(x.size(0), 1, 10, -1)  # Reshaping for 2D pooling

        x = self.frac_maxpool2d(x)

        # Apply Hardswish activation
        x = self.hardswish(x)

        x.share_memory_()  # Sharing memory

        x.tanh_()  # In-place tanh

        x = torch.hypot(x, x)  # Hypotenuse

        x = torch.negative(x)  # Negative

        x = torch.fmin(x, x)  # Element-wise minimum

        x = torch.cos(x)  # Cosine

        x = x.mean(dim=(0, 1)).view(5, 5)

        output1, output2 = torch.lobpcg(x)

        output = output1 * output2

        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted

