import torch
import torch.nn as nn
import torch.nn.functional as F
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
            'torch.Tensor.clone'
            'torch.set_warn_always',
            'torch.nanmean'
            # Removed 'torch.Tensor.mode' from the list as it's not used meaningfully.
        ]

        torch.set_warn_always(True)

    def forward(self, x):
        results = {}
        # Preprocessing
        results['pre_x = x.view(x.size(0), -1)'] = x.clone()
        x = x.view(x.size(0), -1)
        results['x = x.view(x.size(0), -1)'] = x.clone()

        # Step 2: Batch normalization
        results['pre_x = self.LazyBatchNorm1d(x)'] = x.clone()
        x = self.LazyBatchNorm1d(x)
        results['x = self.LazyBatchNorm1d(x)'] = x.clone()

        # Step 3: Reshape
        results['pre_x.view(x.size(0), 3, 32, 32)'] = x.clone()
        x = x.view(x.size(0), 3, 32, 32)
        results['x.view(x.size(0), 3, 32, 32)'] = x.clone()

        # Step 4: Unfold
        results['pre_x = F.unfold(x, kernel_size=(3, 3), padding=1, stride=1)'] = x.clone()
        x = F.unfold(x, kernel_size=(3, 3), padding=1, stride=1)
        results['x = F.unfold(x, kernel_size=(3, 3), padding=1, stride=1)'] = x.clone()

        # Step 5: View
        results['pre_x.view(x.size(0), -1, 32, 32)'] = x.clone()
        x = x.view(x.size(0), -1, 32, 32)
        results['x.view(x.size(0), -1, 32, 32)'] = x.clone()

        # Step 6: Fractional max pooling
        results['pre_x = F.fractional_max_pool2d(x, kernel_size=3, output_ratio=(0.5, 0.5))'] = x.clone()
        x = F.fractional_max_pool2d(x, kernel_size=3, output_ratio=(0.5, 0.5))
        results['x = F.fractional_max_pool2d(x, kernel_size=3, output_ratio=(0.5, 0.5))'] = x.clone()

        # Step 7: Activation function
        results['pre_x = F.elu(x)'] = x.clone()
        x = F.elu(x)
        results['x = F.elu(x)'] = x.clone()

        # Step 8: Operations
        results['pre_x = torch.transpose(x, 1, 3).contiguous()'] = x.clone()
        x = torch.transpose(x, 1, 3).contiguous()
        results['x = torch.transpose(x, 1, 3).contiguous()'] = x.clone()

        results['pre_x = x.view(-1, x.size(2) * x.size(3))'] = x.clone()
        x = x.view(-1, x.size(2) * x.size(3))
        results['x = x.view(-1, x.size(2) * x.size(3))'] = x.clone()

        results['pre_x.frac_()'] = x.clone()
        x.frac_()
        results['x.frac_()'] = x.clone()

        results['pre_x.arctan_()'] = x.clone()
        x.arctan_()
        results['x.arctan_()'] = x.clone()

        results['pre_x.neg_()'] = x.clone()
        x.neg_()
        results['x.neg_()'] = x.clone()

        results['pre_nanmean_result = torch.nanmean(x.view(x.size(0), -1), dim=1)'] = x.clone()
        nanmean_result = torch.nanmean(x.view(x.size(0), -1), dim=1)
        results['nanmean_result = torch.nanmean(x.view(x.size(0), -1), dim=1)'] = nanmean_result.clone()

        results['pre_cloned_x = x.clone()'] = x.clone()
        output = x.clone()
        results['cloned_x = x.clone()'] = output.clone()

        # Finalize the list of used APIs without duplicates and in order of usage
        used_apis_sorted = sorted(set(self.used_apis), key=self.used_apis.index)
        return output, used_apis_sorted, results