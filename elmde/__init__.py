"""
Equilibrium Latent-MoE Decision Engine (ELMDE)
Production Implementation
"""

__version__ = "1.0.0"

from .config import ELMDEConfig
from .nn.model import ELMDEModel
from .conformal.scrc import SelectiveConformalRiskController

__all__ = [
    "ELMDEConfig",
    "ELMDEModel",
    "SelectiveConformalRiskController",
]
