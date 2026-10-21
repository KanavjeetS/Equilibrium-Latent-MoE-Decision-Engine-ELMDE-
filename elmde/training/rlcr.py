from typing import List, Tuple
import torch
import numpy as np

from .losses import StrictlyProperScoringRules


class RLCREngine:
    """
    Reinforcement Learning with Calibration Rewards (RLCR) under Group Relative Policy Optimization (GRPO).
    Computes zero-baseline advantage scores across rollout groups using composite verifiable rewards
    that combine task accuracy with strictly proper Brier and Ranked Probability Scores.
    """
    def __init__(
        self,
        lambda_brier: float = 0.5,
        lambda_rps: float = 0.5,
        eps_adv: float = 1e-8
    ):
        self.lambda_brier = lambda_brier
        self.lambda_rps = lambda_rps
        self.eps_adv = eps_adv

    def compute_grpo_advantages(
        self,
        group_rollout_probs: List[torch.Tensor],
        ground_truth_y: int
    ) -> Tuple[List[float], np.ndarray]:
        """
        Args:
            group_rollout_probs: List of G probability tensors from policy rollouts
            ground_truth_y: Ground truth outcome index
        Returns:
            rewards: List of composite rewards
            advantages: Normalized advantage array of length G with zero mean
        """
        rewards: List[float] = []

        for p in group_rollout_probs:
            pred_y = int(torch.argmax(p).item())
            # 1. Binary task correctness
            r_acc = 1.0 if pred_y == ground_truth_y else 0.0

            # 2. Strictly proper Brier reward
            r_brier = StrictlyProperScoringRules.brier_score(p, ground_truth_y)

            # 3. Strictly proper RPS reward
            r_rps = StrictlyProperScoringRules.ranked_probability_score(p, ground_truth_y)

            # Composite verifiable calibration reward
            r_total = r_acc + self.lambda_brier * r_brier + self.lambda_rps * r_rps
            rewards.append(r_total)

        r_arr = np.array(rewards, dtype=np.float64)
        mean_r = np.mean(r_arr)
        std_r = np.std(r_arr)
        advantages = (r_arr - mean_r) / (std_r + self.eps_adv)
        return rewards, advantages
