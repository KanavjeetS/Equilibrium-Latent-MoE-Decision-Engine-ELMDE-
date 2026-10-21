from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import PreRMSNorm
from .swiglu import SwiGLU
from .mla import MultiHeadLatentAttention
from .equilibrium_core import ContractiveEquilibriumCore
from .heads import HeteroscedasticGlobalChoiceHead, CORNHead, BetaDensityHead
from ..config import ELMDEConfig


class EncoderBlock(nn.Module):
    """Bidirectional Encoder Layer with Pre-RMSNorm, MLA, and SwiGLU."""
    def __init__(self, config: ELMDEConfig):
        super().__init__()
        self.norm1 = PreRMSNorm(config.d_model, eps=config.rms_eps)
        self.mla = MultiHeadLatentAttention(config)
        self.norm2 = PreRMSNorm(config.d_model, eps=config.rms_eps)
        self.swiglu = SwiGLU(config.d_model, config.d_ff)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_kv, k_r = self.mla.compress_kv(x, positions)
        x_norm = self.norm1(x)
        attn_out = self.mla(
            q_input=x_norm,
            c_kv=c_kv,
            k_r=k_r,
            positions=positions,
            attn_mask=attn_mask,
            mode="absorb"
        )
        x_mid = x + attn_out
        ffn_out = self.swiglu(self.norm2(x_mid))
        x_out = x_mid + ffn_out
        return x_out, c_kv, k_r


class ELMDEModel(nn.Module):
    """
    Equilibrium Latent-MoE Decision Engine (ELMDE).
    Unified production model integrating:
    1. Hardware-aligned bidirectional context encoder
    2. Multi-Head Latent Attention (MLA) with Commutative Matrix Absorption
    3. Contractive Recurrent-Depth Sparse MoE Core with Banach fixed-point equilibrium
    4. Tri-Modal Constraint-Manifold Projection Heads (Choice, CORN Ordinal, Beta Density)
    """
    def __init__(self, config: Optional[ELMDEConfig] = None):
        super().__init__()
        self.config = config or ELMDEConfig()

        # Token embedding layer
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)

        # Bidirectional encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(self.config) for _ in range(self.config.n_encoder_layers)
        ])

        # Working memory latent slot generator
        self.slot_embeddings = nn.Parameter(
            torch.randn(self.config.n_latent_slots, self.config.d_model) * 0.02
        )

        # Recurrent MoE Core
        self.equilibrium_core = ContractiveEquilibriumCore(self.config)

        # Tri-modal constraint heads
        self.choice_head = HeteroscedasticGlobalChoiceHead(self.config)
        self.corn_head = CORNHead(self.config)
        self.density_head = BetaDensityHead(self.config)

        # Final RMSNorm
        self.final_norm = PreRMSNorm(self.config.d_model, eps=self.config.rms_eps)

    def encode_context(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes token sequence into contextual representations and extracts persistent prefix MLA memory bank."""
        if positions is None:
            positions = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.long)

        x = self.embedding(input_ids)
        c_kv_last, k_r_last = None, None

        for layer in self.encoder_layers:
            x, c_kv_last, k_r_last = layer(x, positions, attn_mask)

        return x, c_kv_last, k_r_last

    def solve_latent_equilibrium(
        self,
        context_emb: torch.Tensor,
        c_kv: torch.Tensor,
        k_r: torch.Tensor,
        use_ift: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Initializes working memory slots conditioned on context and solves for fixed-point attractor z*."""
        batch_size = context_emb.shape[0]
        pooled_context = context_emb.mean(dim=1, keepdim=True)  # (B, 1, d_model)
        z_init = self.slot_embeddings.unsqueeze(0).expand(batch_size, -1, -1) + pooled_context

        # Contractive recurrent MoE loop
        z_star, core_metrics = self.equilibrium_core(z_init, c_kv, k_r, use_ift=use_ift)

        # Pool across working memory slots into a single decision vector
        z_decision = self.final_norm(z_star.mean(dim=1))  # (B, d_model)
        return z_decision, core_metrics

    def forward_choice(
        self,
        input_ids: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Predicts discrete choice distribution over valid candidates."""
        ctx, c_kv, k_r = self.encode_context(input_ids)
        z_star, metrics = self.solve_latent_equilibrium(ctx, c_kv, k_r)
        probs, tau = self.choice_head(z_star, candidate_embeddings, valid_mask)
        return probs, tau, metrics

    def forward_ordinal(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Predicts rank-consistent ordinal probabilities and cumulative survival curves."""
        ctx, c_kv, k_r = self.encode_context(input_ids)
        z_star, metrics = self.solve_latent_equilibrium(ctx, c_kv, k_r)
        class_probs, cum_survival, cond_probs = self.corn_head(z_star)
        return class_probs, cum_survival, metrics

    def forward_density(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Predicts continuous bounded target mean and variance."""
        ctx, c_kv, k_r = self.encode_context(input_ids)
        z_star, metrics = self.solve_latent_equilibrium(ctx, c_kv, k_r)
        mean, var, alpha, beta = self.density_head(z_star)
        return mean, var, metrics

    def save_pretrained(self, save_directory: str, filename: str = "elmde_model.pt") -> str:
        """Saves model weights and configuration to disk."""
        os.makedirs(save_directory, exist_ok=True)
        filepath = os.path.join(save_directory, filename)
        torch.save({
            "config_dict": asdict(self.config),
            "state_dict": self.state_dict()
        }, filepath)
        return filepath

    @classmethod
    def from_pretrained(cls, filepath: str, device: str = "cpu") -> "ELMDEModel":
        """Loads trained ELMDE weights and configuration from a saved checkpoint."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        if "config_dict" in checkpoint:
            config = ELMDEConfig(**checkpoint["config_dict"])
        elif "config" in checkpoint:
            config = checkpoint["config"]
        else:
            config = ELMDEConfig()

        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model

    @classmethod
    def from_pretrained_backbone(
        cls,
        backbone_name: str = "bert-base-uncased",
        config: Optional[ELMDEConfig] = None
    ) -> "ELMDEModel":
        """
        Bootstraps an ELMDE model by transferring pretrained embedding and encoder representations
        from a Hugging Face bidirectional transformer backbone.
        """
        from transformers import AutoModel
        hf_model = AutoModel.from_pretrained(backbone_name)
        hf_vocab_size = hf_model.config.vocab_size
        hf_hidden_size = hf_model.config.hidden_size

        if config is None:
            config = ELMDEConfig(vocab_size=hf_vocab_size, d_model=hf_hidden_size)
        else:
            config.vocab_size = hf_vocab_size
            config.d_model = hf_hidden_size

        elmde = cls(config)

        # Transfer word embeddings
        with torch.no_grad():
            if hasattr(hf_model, "embeddings") and hasattr(hf_model.embeddings, "word_embeddings"):
                elmde.embedding.weight.copy_(hf_model.embeddings.word_embeddings.weight)

        return elmde
