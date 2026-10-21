from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CORNLoss(nn.Module):
    """
    Conditional Ordinal Regression Loss for CORN head.
    Minimizes conditional binary cross-entropy over nested subsets of samples reaching rank j.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

    def forward(self, cond_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cond_probs: Tensor of shape (batch_size, num_thresholds) containing q_j = sigma(eta_j)
            targets: Ground-truth integer class labels of shape (batch_size,) in {0, ..., num_classes - 1}
        """
        batch_size = targets.shape[0]
        device = targets.device
        total_loss = torch.tensor(0.0, device=device)
        total_subsets = 0

        for j in range(self.num_thresholds):
            # Subset S_j: samples where y >= j
            # For j=0: all samples (S_0)
            if j == 0:
                mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                mask = (targets > (j - 1))

            if mask.any():
                q_j = cond_probs[mask, j]
                # Label is 1 if y > j, 0 if y == j
                y_binary = (targets[mask] > j).float()
                bce = F.binary_cross_entropy(q_j, y_binary, reduction='sum')
                total_loss = total_loss + bce
                total_subsets += mask.sum().item()

        return total_loss / max(total_subsets, 1)


class BetaNLLLoss(nn.Module):
    """
    Exact Negative Log-Likelihood for Continuous Beta Density Head.
    Given alpha, beta > 0 and continuous target y in (0, 1):
    NLL = log Gamma(alpha) + log Gamma(beta) - log Gamma(alpha + beta)
          - (alpha - 1) * log(y) - (beta - 1) * log(1 - y)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Clamp targets strictly inside (0, 1)
        y = torch.clamp(targets, self.eps, 1.0 - self.eps)
        alpha = torch.clamp(alpha, self.eps, 100.0)
        beta = torch.clamp(beta, self.eps, 100.0)

        log_beta_func = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        nll = log_beta_func - (alpha - 1.0) * torch.log(y) - (beta - 1.0) * torch.log(1.0 - y)
        return nll.mean()


class HeteroscedasticLoss(nn.Module):
    """Heteroscedastic Choice Loss with temperature entropy penalty."""
    def __init__(self, tau_reg: float = 0.01):
        super().__init__()
        self.tau_reg = tau_reg

    def forward(self, probs: torch.Tensor, tau: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = torch.log(probs + 1e-9)
        ce = F.nll_loss(log_p, targets)
        reg = self.tau_reg * torch.mean(torch.abs(tau - 1.0))
        return ce + reg


class StrictlyProperScoringRules:
    """Exact implementations of strictly proper scoring rules for probability calibration."""

    @staticmethod
    def brier_score(probs: torch.Tensor, target_idx: int) -> float:
        """Bounded strictly proper quadratic utility: 1 - ||p - e_y||^2 in [0, 1]."""
        p = probs.detach().cpu().numpy()
        e_y = 0.0
        sum_sq = 0.0
        for i, val in enumerate(p):
            target_val = 1.0 if i == target_idx else 0.0
            sum_sq += (val - target_val) ** 2
        return 1.0 - sum_sq

    @staticmethod
    def log_score(probs: torch.Tensor, target_idx: int, eps: float = 1e-12) -> float:
        """Local strictly proper logarithmic utility: log(p_y)."""
        p_y = float(probs[target_idx].item())
        import math
        return math.log(max(p_y, eps))

    @staticmethod
    def spherical_score(probs: torch.Tensor, target_idx: int, eps: float = 1e-12) -> float:
        """Non-local spherical utility: p_y / ||p||_2."""
        norm = torch.norm(probs, p=2).item()
        p_y = float(probs[target_idx].item())
        return p_y / max(norm, eps)

    @staticmethod
    def ranked_probability_score(probs: torch.Tensor, target_idx: int) -> float:
        """
        Ranked Probability Score (RPS) for ordinal calibration.
        Measures integrated distance between cumulative forecast and ground-truth step function.
        """
        cum_p = torch.cumsum(probs, dim=-1).detach().cpu().numpy()
        k = len(cum_p)
        penalty = 0.0
        for m in range(k - 1):
            indicator = 1.0 if target_idx <= m else 0.0
            penalty += (cum_p[m] - indicator) ** 2
        return -penalty / max(k - 1, 1)
