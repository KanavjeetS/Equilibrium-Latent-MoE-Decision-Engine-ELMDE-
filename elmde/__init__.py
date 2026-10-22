"""
Equilibrium Latent-MoE Decision Engine (ELMDE)
Non-Autoregressive Decision Intelligence with Exact Manifold Projection & Conformal Risk Guarantees.

A plug-and-play system feature that can be injected into ANY AI / LLM pipeline,
agent loop, LangChain/LlamaIndex chain, or PyTorch workflow to replace slow,
distorted autoregressive token-by-token constrained decoding with exact,
calibrated manifold projections in <95ms with certified safety.
"""

__version__ = "2.0.0"

from .config import ELMDEConfig
from .nn.model import ELMDEModel
from .conformal.scrc import SelectiveConformalRiskController
from .injector import ELMDE, DecisionResult, TriageResult, ContinuousEstimate

__all__ = [
    "ELMDE",
    "DecisionResult",
    "TriageResult",
    "ContinuousEstimate",
    "ELMDEConfig",
    "ELMDEModel",
    "SelectiveConformalRiskController",
]
