"""
Foundation Backbone Integration for ELMDE (Equilibrium Latent-MoE Decision Engine).
Connects pre-trained ModernBERT bidirectional contextual representations (2T tokens)
directly to the Contractive Recurrent-Depth MoE Core and Manifold Projection Heads.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .rms_norm import PreRMSNorm
from .mla import MultiHeadLatentAttention
from .equilibrium_core import ContractiveEquilibriumCore
from .heads import HeteroscedasticGlobalChoiceHead, CORNHead, BetaDensityHead
from ..config import ELMDEConfig


class ELMDEFoundationModel(nn.Module):
    """
    ELMDE Foundation Model:
    Fuses pre-trained bidirectional foundation representations (e.g. ModernBERT)
    with the Recurrent-Depth Sparse MoE Equilibrium Core and Manifold Projection Heads.
    
    Eliminates autoregressive decoding while preserving deep foundation-level world knowledge.
    """
    def __init__(
        self,
        backbone_name: str = "answerdotai/ModernBERT-base",
        config: Optional[ELMDEConfig] = None,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(backbone_name)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        hidden_size = self.backbone.config.hidden_size  # 768 for ModernBERT-base
        
        # Configure ELMDE core to match backbone dimension
        if config is None:
            self.config = ELMDEConfig(
                d_model=hidden_size,
                d_ff=2048,
                n_heads=12,
                d_head=64,
                d_kv_latent=128,
                d_q_latent=256,
                d_rope=32,
                n_latent_slots=8,
                n_experts=8,
                top_k_experts=2,
                max_recurrent_steps=6,
                contraction_alpha=0.45,
                halt_tol=1e-3,
            )
        else:
            self.config = config
            self.config.d_model = hidden_size
            
        # Multi-Head Latent Attention working over backbone contextual states
        self.mla = MultiHeadLatentAttention(self.config)
        self.norm_context = PreRMSNorm(self.config.d_model, eps=self.config.rms_eps)
        
        # Working memory latent slot generator
        self.slot_embeddings = nn.Parameter(
            torch.randn(self.config.n_latent_slots, self.config.d_model) * 0.02
        )
        
        # Contractive Recurrent-Depth MoE Core (O(1) memory via IFT)
        self.equilibrium_core = ContractiveEquilibriumCore(self.config)
        
        # Manifold Decision Heads
        self.choice_head = HeteroscedasticGlobalChoiceHead(self.config)
        self.corn_head = CORNHead(self.config)
        self.density_head = BetaDensityHead(self.config)
        
        # Final LayerNorm
        self.final_norm = PreRMSNorm(self.config.d_model, eps=self.config.rms_eps)

    def forward_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extracts contextual token representations from ModernBERT."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state  # (B, L, d_model)

    def solve_latent_equilibrium(
        self,
        context_emb: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        use_ift: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compresses contextual representations into MLA memory bank, initializes
        working memory slots, and iterates to fixed-point attractor z*.
        """
        B, L, D = context_emb.shape
        if positions is None:
            positions = torch.arange(L, device=context_emb.device, dtype=torch.long)
            
        # Compress into Latent KV bank
        context_norm = self.norm_context(context_emb)
        c_kv, k_r = self.mla.compress_kv(context_norm, positions)
        
        # Condition initial working memory slots on mean-pooled context
        pooled_context = context_emb.mean(dim=1, keepdim=True)
        z_init = self.slot_embeddings.unsqueeze(0).expand(B, -1, -1) + pooled_context
        
        # Solve fixed-point recurrence: z_{t+1} = (1-lambda) z_t + lambda Psi(z_t, c_kv, k_r)
        z_star, core_metrics = self.equilibrium_core(z_init, c_kv, k_r, use_ift=use_ift)
        
        # Pool across working memory slots into unified decision vector
        z_decision = self.final_norm(z_star.mean(dim=1))
        return z_decision, core_metrics

    def forward_ordinal(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_ift: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Predicts monotonic rank-consistent ordinal probabilities."""
        ctx = self.forward_backbone(input_ids, attention_mask)
        z_star, metrics = self.solve_latent_equilibrium(ctx, use_ift=use_ift)
        class_probs, cum_survival, cond_probs = self.corn_head(z_star)
        return class_probs, cum_survival, metrics

    def forward_choice(
        self,
        input_ids: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        use_ift: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Predicts unwarped global choice probabilities with heteroscedastic temperature."""
        ctx = self.forward_backbone(input_ids, attention_mask)
        z_star, metrics = self.solve_latent_equilibrium(ctx, use_ift=use_ift)
        probs, tau = self.choice_head(z_star, candidate_embeddings, valid_mask)
        return probs, tau, metrics

    def forward_density(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_ift: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Predicts continuous Beta density parameters and moments."""
        ctx = self.forward_backbone(input_ids, attention_mask)
        z_star, metrics = self.solve_latent_equilibrium(ctx, use_ift=use_ift)
        mean, var, alpha, beta = self.density_head(z_star)
        return mean, var, metrics

    def save_pretrained(self, save_directory: str, filename: str = "elmde_foundation.pt") -> str:
        """Saves model state and config."""
        os.makedirs(save_directory, exist_ok=True)
        filepath = os.path.join(save_directory, filename)
        torch.save({
            "backbone_name": self.backbone_name,
            "config_dict": asdict(self.config),
            "state_dict": self.state_dict()
        }, filepath)
        return filepath

    @classmethod
    def from_pretrained(
        cls,
        filepath: str,
        device: str = "cpu",
        freeze_backbone: bool = False
    ) -> "ELMDEFoundationModel":
        """Loads saved ELMDEFoundationModel checkpoint."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        backbone_name = checkpoint.get("backbone_name", "answerdotai/ModernBERT-base")
        config_dict = checkpoint.get("config_dict", {})
        config = ELMDEConfig(**config_dict)
        
        model = cls(backbone_name=backbone_name, config=config, freeze_backbone=freeze_backbone)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model
