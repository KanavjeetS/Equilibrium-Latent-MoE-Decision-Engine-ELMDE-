from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ELMDEConfig:
    """
    Hardware-aligned configuration for the Equilibrium Latent-MoE Decision Engine (ELMDE).
    All primary hidden, feedforward, and head dimensions are multiples of 64/128
    for optimal Tensor Core alignment and zero-padding jagged sequence execution.
    """
    # Model dimensions
    vocab_size: int = 50368          # 787 * 64 (ModernBERT hardware-aligned vocabulary)
    d_model: int = 768               # 12 * 64 hidden state dimension
    n_heads: int = 12                # Multi-Head Latent Attention heads (H)
    d_head: int = 64                 # Content head dimension (d_h)
    d_rope: int = 32                 # Decoupled RoPE subspace dimension (d_r)
    d_kv_latent: int = 128           # Compressed KV latent dimension (d_kv << H * d_h)
    d_q_latent: int = 256            # Compressed Query latent dimension (d_q)
    d_ff: int = 2048                 # SwiGLU hidden dimension (multiple of 64)

    # Attention & Encoder configuration
    n_encoder_layers: int = 6        # Total bidirectional encoder layers
    local_window_size: int = 128     # Sliding window size for local layers (l % 3 != 0)
    rope_theta_global: float = 160000.0  # Base wavelength for global RoPE
    rope_theta_local: float = 10000.0    # Base wavelength for local RoPE
    rms_eps: float = 1e-6            # RMSNorm numerical stability epsilon
    dropout: float = 0.0

    # Recurrent MoE Core configuration
    n_latent_slots: int = 8          # Working memory latent slots (M)
    n_experts: int = 8               # Total routed experts (E)
    top_k_experts: int = 2           # Active routed experts per token (K)
    max_recurrent_steps: int = 6     # Maximum recurrence depth (R_max)
    contraction_alpha: float = 0.45  # Fixed-point damping coefficient in (0, 1)
    halt_tol: float = 1e-3           # Relative residual threshold for adaptive halting
    aux_loss_weight: float = 0.01    # Router load-balancing auxiliary loss weight

    # Tri-Modal Constraint Heads
    num_ordinal_classes: int = 5     # Ordered ranks for CORN head
    tau_min: float = 0.1             # Minimum heteroscedastic temperature
    tau_max: float = 5.0             # Maximum heteroscedastic temperature

    # Selective Conformal Risk Control (SCRC)
    scrc_alpha_target: float = 0.01  # Certified error ceiling (99% accuracy floor)
    scrc_loss_bound_b: float = 1.0   # Bounded loss upper limit
