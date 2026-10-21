import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import PreRMSNorm
from .rope import DecoupledRoPE
from ..config import ELMDEConfig


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) with Decoupled RoPE & Commutative Matrix Absorption.
    Compresses Key-Value cache into a shared low-rank latent subspace,
    and supports zero-materialization Absorb Mode during inference and iterative recurrent loops.
    """
    def __init__(self, config: ELMDEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_rope = config.d_rope
        self.d_kv_latent = config.d_kv_latent
        self.d_q_latent = config.d_q_latent

        # Latent KV compression
        self.w_dkv = nn.Linear(self.d_model, self.d_kv_latent, bias=False)
        self.norm_kv = PreRMSNorm(self.d_kv_latent, eps=config.rms_eps)

        # Latent Query compression
        self.w_dq = nn.Linear(self.d_model, self.d_q_latent, bias=False)
        self.norm_q = PreRMSNorm(self.d_q_latent, eps=config.rms_eps)

        # Content up-projections
        self.w_uk = nn.Linear(self.d_kv_latent, self.n_heads * self.d_head, bias=False)
        self.w_uv = nn.Linear(self.d_kv_latent, self.n_heads * self.d_head, bias=False)
        self.w_uq = nn.Linear(self.d_q_latent, self.n_heads * self.d_head, bias=False)

        # Decoupled RoPE projections
        self.w_qr = nn.Linear(self.d_q_latent, self.n_heads * self.d_rope, bias=False)
        self.w_kr = nn.Linear(self.d_model, self.d_rope, bias=False)
        self.rope_global = DecoupledRoPE(self.d_rope, theta_base=config.rope_theta_global)

        # Output projection
        self.w_o = nn.Linear(self.n_heads * self.d_head, self.d_model, bias=False)

        # Scale factor
        self.scale = 1.0 / math.sqrt(self.d_head + self.d_rope)

    def compress_kv(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compresses input representations into the persistent MLA KV memory bank.
        Returns:
            c_kv: (batch, seq_len, d_kv_latent)
            k_r:  (batch, seq_len, d_rope)
        """
        c_kv = self.norm_kv(self.w_dkv(x))
        k_r_unrotated = self.w_kr(x)
        k_r = self.rope_global(k_r_unrotated, positions)
        return c_kv, k_r

    def forward(
        self,
        q_input: torch.Tensor,
        c_kv: torch.Tensor,
        k_r: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        mode: str = "absorb"
    ) -> torch.Tensor:
        """
        Args:
            q_input: Query tensor of shape (batch, q_len, d_model)
            c_kv: Compressed KV latent of shape (batch, kv_len, d_kv_latent)
            k_r: Decoupled RoPE Key of shape (batch, kv_len, d_rope)
            positions: Token positions of shape (q_len,)
            attn_mask: Optional attention mask
            mode: 'absorb' or 'decompress'
        """
        b_size, q_len, _ = q_input.shape
        kv_len = c_kv.shape[1]

        # Compress Queries
        c_q = self.norm_q(self.w_dq(q_input))  # (B, Q, d_q)

        # Decoupled RoPE Queries: reshape to (B, H, Q, d_rope)
        q_r_unrotated = self.w_qr(c_q).view(b_size, q_len, self.n_heads, self.d_rope).permute(0, 2, 1, 3)
        q_r = self.rope_global(q_r_unrotated, positions)  # (B, H, Q, d_rope)

        # RoPE attention score component: (B, H, Q, d_rope) @ (B, 1, d_rope, kv_len) -> (B, H, Q, kv_len)
        scores_rope = torch.matmul(q_r, k_r.unsqueeze(1).transpose(-1, -2))

        if mode == "absorb":
            # --- ABSORB MODE ---
            # W_uq: (H * d_h, d_q) -> (H, d_h, d_q)
            w_uq_head = self.w_uq.weight.view(self.n_heads, self.d_head, self.d_q_latent)
            # W_uk: (H * d_h, d_kv) -> (H, d_h, d_kv)
            w_uk_head = self.w_uk.weight.view(self.n_heads, self.d_head, self.d_kv_latent)

            # w_ukq = w_uq^T @ w_uk: (H, d_q, d_kv)
            w_ukq = torch.bmm(w_uq_head.transpose(1, 2), w_uk_head)

            # q_abs = c_q @ w_ukq: (B, 1, Q, d_q) @ (1, H, d_q, d_kv) -> (B, H, Q, d_kv)
            q_abs = torch.matmul(c_q.unsqueeze(1), w_ukq.unsqueeze(0))

            # Content score = q_abs @ c_kv^T: (B, H, Q, d_kv) @ (B, 1, d_kv, kv_len) -> (B, H, Q, kv_len)
            scores_content = torch.matmul(q_abs, c_kv.unsqueeze(1).transpose(-1, -2))

            # Total attention logits
            scores = (scores_content + scores_rope) * self.scale

            if attn_mask is not None:
                scores = scores + attn_mask

            attn_weights = F.softmax(scores, dim=-1)  # (B, H, Q, kv_len)

            # Head output absorbs W_uv and W_o:
            # W_uv: (H * d_h, d_kv) -> (H, d_kv, d_h)
            w_uv_head = self.w_uv.weight.view(self.n_heads, self.d_head, self.d_kv_latent).transpose(1, 2)
            # W_o: (d_model, H * d_h) -> (H, d_h, d_model)
            w_o_head = self.w_o.weight.view(self.d_model, self.n_heads, self.d_head).permute(1, 2, 0)
            # W_uvo: (H, d_kv, d_model)
            w_uvo = torch.bmm(w_uv_head, w_o_head)

            # Latent context = attn_weights @ c_kv: (B, H, Q, kv_len) @ (B, 1, kv_len, d_kv) -> (B, H, Q, d_kv)
            latent_context = torch.matmul(attn_weights, c_kv.unsqueeze(1))

            # Map through absorbed output projection: (B, H, Q, d_kv) @ (1, H, d_kv, d_model) -> (B, H, Q, d_model)
            out_per_head = torch.matmul(latent_context, w_uvo.unsqueeze(0))
            output = out_per_head.sum(dim=1)  # (B, Q, d_model)
            return output

        else:
            # --- EXPLICIT DECOMPRESSION MODE ---
            q_c = self.w_uq(c_q).view(b_size, q_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            k_c = self.w_uk(c_kv).view(b_size, kv_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            v_c = self.w_uv(c_kv).view(b_size, kv_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)

            q_full = torch.cat([q_c, q_r], dim=-1)  # (B, H, Q, d_h + d_r)
            k_r_exp = k_r.unsqueeze(1).expand(b_size, self.n_heads, kv_len, self.d_rope)
            k_full = torch.cat([k_c, k_r_exp], dim=-1)  # (B, H, kv_len, d_h + d_r)

            scores = torch.matmul(q_full, k_full.transpose(-1, -2)) * self.scale
            if attn_mask is not None:
                scores = scores + attn_mask
            attn_weights = F.softmax(scores, dim=-1)

            context = torch.matmul(attn_weights, v_c)  # (B, H, Q, d_h)
            context_flat = context.permute(0, 2, 1, 3).contiguous().view(b_size, q_len, self.n_heads * self.d_head)
            return self.w_o(context_flat)
