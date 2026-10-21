import torch
import torch.nn as nn


class DecoupledRoPE(nn.Module):
    """
    Decoupled Rotary Position Embedding (RoPE) for Multi-Head Latent Attention.
    Rotates only the dedicated positional subspace of dimension d_r,
    leaving the content head subspace d_h unrotated and linearly separable.
    """
    def __init__(self, d_rope: int, theta_base: float = 10000.0):
        super().__init__()
        assert d_rope % 2 == 0, f"d_rope must be even, got {d_rope}"
        self.d_rope = d_rope
        self.theta_base = theta_base

        # Precompute inverse frequencies: theta_i = base^(-2k / d_rope)
        indices = torch.arange(0, d_rope, 2, dtype=torch.float32)
        inv_freq = theta_base ** (-indices / d_rope)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape (..., seq_len, d_rope)
            positions: 1D Tensor of token positions with shape (seq_len,)
        """
        # pos_float: (seq_len,)
        pos_float = positions.to(self.inv_freq.dtype)
        # angles: (seq_len, d_rope / 2)
        angles = torch.outer(pos_float, self.inv_freq)
        cos_a = torch.cos(angles).to(x.dtype)
        sin_a = torch.sin(angles).to(x.dtype)

        # Reshape cos_a to broadcast with x: (1, ..., 1, seq_len, d_rope / 2)
        # Leading dimensions: all dims of x except the last two
        lead_ones = [1] * (x.ndim - 2)
        cos_a = cos_a.view(*lead_ones, positions.shape[0], self.d_rope // 2)
        sin_a = sin_a.view(*lead_ones, positions.shape[0], self.d_rope // 2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos_a - x_odd * sin_a
        out[..., 1::2] = x_even * sin_a + x_odd * cos_a
        return out
