import torch
from torch import nn


class RTDLoss(nn.Module):
    def forward(self, pred_rtd: torch.Tensor, target_rtd: torch.Tensor):
        """
        Remaining Track Distance Error (RTD) loss. Can be used with either the raw trajectory distance or the RTD (including distance to threshold).
        """
        diff_rtd = (pred_rtd - target_rtd).abs()  # [batch_size]
        relative_rtd_error = diff_rtd / target_rtd # [batch_size]
        return relative_rtd_error.mean()