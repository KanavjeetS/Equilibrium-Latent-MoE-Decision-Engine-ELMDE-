from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ELMDEConfig


class HeteroscedasticGlobalChoiceHead(nn.Module):
    """
    Discrete Constraint Satisfaction Head.
    Scores complete candidate vectors globally in a single forward pass,
    completely eliminating autoregressive prefix-sharing distortion (D_KL = 0).
    Predicts an input-dependent heteroscedastic temperature tau_phi(z*).
    """
    def __init__(self, config: ELMDEConfig):
        super().__init__()
        self.d_model = config.d_model
        self.tau_min = config.tau_min
        self.tau_max = config.tau_max

        # Candidate alignment projection
        self.w_choice = nn.Linear(self.d_model, self.d_model, bias=False)

        # Heteroscedastic temperature predictor
        self.tau_predictor = nn.Linear(self.d_model, 1, bias=True)

    def forward(
        self,
        z_pooled: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_pooled: Latent decision state (batch_size, d_model)
            candidate_embeddings: Candidate embeddings (batch_size, num_candidates, d_model)
            valid_mask: Boolean tensor of valid candidates (batch_size, num_candidates)
        Returns:
            probs: Exact probability distribution over candidates (batch_size, num_candidates)
            tau: Learned heteroscedastic temperature (batch_size, 1)
        """
        # Learned temperature tau in [tau_min, tau_max]
        tau_raw = torch.sigmoid(self.tau_predictor(z_pooled))
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_raw  # (B, 1)

        # Projected query
        z_proj = self.w_choice(z_pooled).unsqueeze(1)  # (B, 1, d_model)

        # Dot-product logits scaled by sqrt(d) and heteroscedastic temperature
        scale = 1.0 / (torch.sqrt(torch.tensor(float(self.d_model), device=z_pooled.device)) * tau.unsqueeze(-1))
        logits = torch.sum(z_proj * candidate_embeddings, dim=-1, keepdim=False) * scale.squeeze(-1)  # (B, C)

        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e9)

        probs = F.softmax(logits, dim=-1)
        return probs, tau


class CORNHead(nn.Module):
    """
    Conditional Ordinal Regression for Neural Networks (CORN) Head.
    Predicts K-1 conditional probabilities q_j = P(Y > j | Y > j-1) = sigmoid(w_j^T z* + b_j).
    Guarantees strict rank monotonicity across all thresholds without weight-tying restrictions.
    """
    def __init__(self, config: ELMDEConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_classes = config.num_ordinal_classes
        self.num_thresholds = self.num_classes - 1

        # K-1 independent threshold classifiers
        self.fc = nn.Linear(self.d_model, self.num_thresholds, bias=True)

    def forward(
        self, z_pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_pooled: Latent decision state (batch_size, d_model)
        Returns:
            class_probs: Exact categorical distribution over K classes (batch_size, K)
            cum_survival: Unconditional cumulative survival probabilities P(Y > k) (batch_size, K-1)
            cond_probs: Conditional exceedance probabilities q_j (batch_size, K-1)
        """
        logits = self.fc(z_pooled)  # (B, K-1)
        cond_probs = torch.sigmoid(logits)  # (B, K-1), q_j in (0, 1)

        # Cumulative survival probabilities: P(Y > k) = prod_{j=0}^k q_j
        cum_survival = torch.cumprod(cond_probs, dim=-1)  # (B, K-1)

        # Analytical class distribution
        batch_size = z_pooled.shape[0]
        class_probs = torch.zeros(batch_size, self.num_classes, device=z_pooled.device, dtype=z_pooled.dtype)

        # P(Y = 0) = 1 - q_0
        class_probs[:, 0] = 1.0 - cond_probs[:, 0]

        # P(Y = k) = (1 - q_k) * prod_{j=0}^{k-1} q_j = (1 - q_k) * cum_survival[k-1]
        for k in range(1, self.num_thresholds):
            class_probs[:, k] = (1.0 - cond_probs[:, k]) * cum_survival[:, k - 1]

        # P(Y = K-1) = prod_{j=0}^{K-2} q_j = cum_survival[K-2]
        class_probs[:, -1] = cum_survival[:, -1]

        # Clean numerical precision clamp
        class_probs = torch.clamp(class_probs, 0.0, 1.0)
        class_probs = class_probs / class_probs.sum(dim=-1, keepdim=True)

        return class_probs, cum_survival, cond_probs


class BetaDensityHead(nn.Module):
    """
    Continuous Bounded Density Head.
    Parameterizes a Beta(alpha, beta) distribution over normalized continuous targets in (0, 1).
    Outputs exact closed-form epistemic mean and aleatoric predictive variance.
    """
    def __init__(self, config: ELMDEConfig):
        super().__init__()
        self.d_model = config.d_model
        self.proj_alpha = nn.Linear(self.d_model, 1, bias=True)
        self.proj_beta = nn.Linear(self.d_model, 1, bias=True)

    def forward(
        self, z_pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: Expected value alpha / (alpha + beta)
            variance: Predictive uncertainty
            alpha: Softplus-bounded alpha parameter
            beta: Softplus-bounded beta parameter
        """
        alpha = 1.0 + F.softplus(self.proj_alpha(z_pooled))
        beta = 1.0 + F.softplus(self.proj_beta(z_pooled))

        mean = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta).pow(2) * (alpha + beta + 1.0))
        return mean, var, alpha, beta
