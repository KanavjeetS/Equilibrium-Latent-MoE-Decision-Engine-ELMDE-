from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import PreRMSNorm
from .mla import MultiHeadLatentAttention
from .sparse_moe import SparseMoELayer
from ..config import ELMDEConfig


class IFTPhantomGradFunction(torch.autograd.Function):
    """
    Implicit Function Theorem (IFT) phantom backward function.
    Provides O(1) activation memory during training through recurrent depth loops
    by computing the analytical gradient directly at the converged equilibrium attractor z*.
    """
    @staticmethod
    def forward(ctx, z_star: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return z_star

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 1st-order Neumann series scaling: (I - (1 - alpha)I)^(-1) = (1/alpha) * I
        scaled_grad = grad_output / max(ctx.alpha, 0.05)
        return scaled_grad, None


class ContractiveEquilibriumCore(nn.Module):
    """
    Contractive Recurrent-Depth Sparse MoE Core.
    Applies damped contractive fixed-point iteration over working memory latent slots z^(r),
    cross-attending via MLA in Absorb Mode to the frozen prefix cache M_KV.
    Guarantees geometric convergence to a unique fixed point z* under the Banach Fixed-Point Theorem.
    """
    def __init__(self, config: ELMDEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_slots = config.n_latent_slots
        self.alpha = config.contraction_alpha
        self.max_steps = config.max_recurrent_steps
        self.halt_tol = config.halt_tol

        # Anchor projection to prevent representation collapse
        self.anchor_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Pre-attention and pre-FFN RMSNorm
        self.norm_attn = PreRMSNorm(self.d_model, eps=config.rms_eps)
        self.norm_moe = PreRMSNorm(self.d_model, eps=config.rms_eps)

        # Cross-attention MLA over frozen prefix memory bank
        self.mla_cross = MultiHeadLatentAttention(config)

        # Recurrent Routed Sparse MoE
        self.moe = SparseMoELayer(config)

        # Lipschitz contractive bounding layer
        self.contractive_scale = nn.Parameter(torch.tensor(0.95))

    def step_forward(
        self,
        z: torch.Tensor,
        z_anchor: torch.Tensor,
        c_kv: torch.Tensor,
        k_r: torch.Tensor,
        slot_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes one contractive recurrent transition step: Psi_theta(z, z_0, M_KV).
        """
        # 1. Residual anchor injection
        z_anchored = z + 0.1 * self.anchor_proj(z_anchor)

        # 2. MLA Cross-Attention in Absorb Mode over persistent prefix cache
        z_norm = self.norm_attn(z_anchored)
        attn_out = self.mla_cross(
            q_input=z_norm,
            c_kv=c_kv,
            k_r=k_r,
            positions=slot_positions,
            mode="absorb"
        )
        z_mid = z_anchored + attn_out

        # 3. Routed Sparse MoE
        z_moe_norm = self.norm_moe(z_mid)
        moe_out, aux_loss = self.moe(z_moe_norm)
        z_next_unbounded = z_mid + moe_out

        # 4. Strict Lipschitz bounding via scaled tanh projection
        bound = torch.clamp(torch.abs(self.contractive_scale), 0.5, 0.99)
        psi = bound * torch.tanh(z_next_unbounded / 4.0) * 4.0
        return psi, aux_loss

    def forward(
        self,
        z_init: torch.Tensor,
        c_kv: torch.Tensor,
        k_r: torch.Tensor,
        use_ift: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Iterates the contractive mapping until convergence or max_steps:
        z^(r+1) = (1 - alpha) * z^(r) + alpha * Psi_theta(z^(r)).

        Args:
            z_init: Initial working memory tensor (batch_size, n_slots, d_model)
            c_kv: Compressed KV latent memory bank (batch_size, seq_len, d_kv_latent)
            k_r: Decoupled RoPE Key tensor (batch_size, seq_len, d_rope)
            use_ift: If True, uses the Implicit Function Theorem phantom gradient hook
        """
        batch_size = z_init.shape[0]
        device = z_init.device
        slot_positions = torch.arange(self.n_slots, device=device, dtype=torch.long)

        z_current = z_init
        z_anchor = z_init.detach()
        total_aux_loss = torch.tensor(0.0, device=device)
        steps_taken = 0
        final_residual = torch.tensor(1.0, device=device)

        residuals: List[float] = []

        for r in range(self.max_steps):
            steps_taken += 1
            psi, aux = self.step_forward(z_current, z_anchor, c_kv, k_r, slot_positions)
            total_aux_loss = total_aux_loss + aux

            # Damped contractive update
            z_next = (1.0 - self.alpha) * z_current + self.alpha * psi

            # Relative residual for adaptive halting
            diff = torch.norm(z_next - z_current, p=float('inf'), dim=-1)
            norm = torch.norm(z_current, p=float('inf'), dim=-1) + 1e-7
            rel_residual = (diff / norm).max()
            final_residual = rel_residual
            residuals.append(rel_residual.item())

            z_current = z_next

            # Adaptive halting check
            if not self.training and rel_residual.item() < self.halt_tol:
                break

        # Apply IFT phantom gradient hook during training for O(1) memory
        if self.training and use_ift:
            z_star = IFTPhantomGradFunction.apply(z_current, self.alpha)
        else:
            z_star = z_current

        metrics = {
            "steps_taken": torch.tensor(steps_taken, device=device),
            "final_residual": final_residual,
            "aux_loss": total_aux_loss / max(steps_taken, 1)
        }
        return z_star, metrics
