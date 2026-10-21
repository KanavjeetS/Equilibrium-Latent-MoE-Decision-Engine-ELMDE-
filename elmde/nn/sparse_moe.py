from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .swiglu import SwiGLU
from ..config import ELMDEConfig


class SparseMoELayer(nn.Module):
    """
    Hardware-aligned Routed Sparse Mixture-of-Experts (MoE) Layer.
    Implements:
    - 1 shared always-active expert E_0 (DeepSeekMoE design)
    - E routed SwiGLU experts with Top-K dynamic token gating
    - Vectorized token grouping and expert dispatch
    - Differentiable auxiliary load-balancing loss computation
    """
    def __init__(self, config: ELMDEConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.n_experts = config.n_experts
        self.top_k = config.top_k_experts
        self.aux_loss_weight = config.aux_loss_weight

        # 1. Shared always-active expert
        self.shared_expert = SwiGLU(self.d_model, self.d_ff)

        # 2. Routed experts
        self.routed_experts = nn.ModuleList([
            SwiGLU(self.d_model, self.d_ff) for _ in range(self.n_experts)
        ])

        # 3. Router gating projection (de-biased)
        self.router = nn.Linear(self.d_model, self.n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (..., d_model)
        Returns:
            out: Expert output of identical shape (..., d_model)
            aux_loss: Load balancing auxiliary loss scalar
        """
        original_shape = x.shape
        x_flat = x.view(-1, self.d_model)  # (N, d_model)
        num_tokens = x_flat.shape[0]

        # 1. Shared expert contribution (vectorized)
        shared_out = self.shared_expert(x_flat)

        # 2. Router logits & probabilities
        router_logits = self.router(x_flat)  # (N, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, n_experts)

        # 3. Top-K expert selection
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (N, K)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)  # Renormalize

        # 4. Vectorized expert dispatch & aggregation
        routed_out = torch.zeros_like(x_flat)

        # Compute token mask per expert across all K slots simultaneously
        for e in range(self.n_experts):
            # Check which (token, k) pairs select expert e
            mask = (topk_indices == e)  # (N, K)
            if mask.any():
                token_indices, k_slots = torch.where(mask)
                tokens_e = x_flat[token_indices]
                weights_e = topk_weights[token_indices, k_slots].unsqueeze(-1)
                expert_out_e = self.routed_experts[e](tokens_e)
                routed_out.index_add_(0, token_indices, weights_e * expert_out_e)

        # 5. Differentiable Load Balancing Auxiliary Loss
        # f_e: empirical fraction of tokens routed to expert e
        expert_mask = F.one_hot(topk_indices[:, 0], num_classes=self.n_experts).float()
        f_e = expert_mask.mean(dim=0)  # (n_experts,)
        P_e = router_probs.mean(dim=0)  # (n_experts,)
        aux_loss = self.aux_loss_weight * self.n_experts * torch.sum(f_e * P_e)

        # Total output = shared expert + routed experts
        total_out = (shared_out + routed_out).view(original_shape)
        return total_out, aux_loss
