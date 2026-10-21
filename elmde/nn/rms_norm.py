import torch
import torch.nn as nn


class PreRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Pre-RMSNorm).
    Eliminates mean-centering to save one statistical reduction per layer
    while maintaining training stability and GPU Tensor Core alignment.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        variance = x.pow(2).mean(-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        return norm_x * self.weight
