import torch
import torch.nn as nn
import torch.nn.functional as F

class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.LazyBatchNorm1d = nn.LazyBatchNorm1d(0)  # 0 is a placeholder
        self.placeholder_layer = nn.Linear(8 * 8 * 3, 100)


class PreprocessAndCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.LazyBatchNorm1d = nn.LazyBatchNorm1d(0)  # 0 is a placeholder

        self.used_apis = [
            'torch.nn.LazyBatchNorm1d',
            'torch.Tensor.size',
            'torch.Tensor.view',
            'torch.nn.functional.unfold',
            'torch.nn.functional.fractional_max_pool2d',
            'torch.nn.functional.elu',
            'torch.transpose',
            'torch.Tensor.frac_',
            'torch.Tensor.arctan_',
            'torch.Tensor.neg',
            'torch.set_warn_always',
            'torch.nanmean'
        ]

        torch.set_warn_always(True)

    def forward(self, x):
        # Preprocessing
        x = x.view(x.size(0), -1)

        # Step 2: Batch normalization
        x = self.LazyBatchNorm1d(x)
        # Step 3: Reshape
        x = x.view(x.size(0), 3, 32, 32)

        # Step 4: Unfold
        x = F.unfold(x, kernel_size=(3, 3), padding=1, stride=1)

        # Step 5: View
        x = x.view(x.size(0), -1, 32, 32)

        # Step 6: Fractional max pooling
        x = F.fractional_max_pool2d(x, kernel_size=3, output_ratio=(0.5, 0.5))
        # Step 7: Activation function
        x = F.elu(x)
        # Step 8: Operations
        x = torch.transpose(x, 1, 3).contiguous()

        x = x.view(-1, x.size(2) * x.size(3))

        x.frac_()

        x.arctan_()

        x.neg_()

        nanmean_result = torch.nanmean(x.view(x.size(0), -1), dim=1)

        output = nanmean_result.view(-1, 1)

        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)

        return output, used_apis_sorted