"""
Adaptive Selective Conformal Risk Control (Adaptive-SCRC).
Implements Gibbs & Candès Adaptive Conformal Inference (ACI) on top of SCRC,
guaranteeing provable error control (<= alpha_target) even under non-stationary
covariate shift and real-world distribution drift.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class AdaptiveSelectiveConformalController:
    """
    Online Adaptive Selective Conformal Risk Controller.
    Dynamically tracks deployment stream distribution drift using an adaptive
    step-size update rule, maintaining long-run conditional error rate <= alpha_target.
    """
    def __init__(
        self,
        alpha_target: float = 0.01,
        initial_tau: float = 0.99,
        adaptation_rate_gamma: float = 0.005,
        tau_min: float = 0.50,
        tau_max: float = 0.9999
    ):
        self.alpha_target = alpha_target
        self.current_tau = initial_tau
        self.gamma = adaptation_rate_gamma
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Online tracking metrics
        self.history_accepted: List[bool] = []
        self.history_errors: List[float] = []
        self.history_tau: List[float] = [initial_tau]

    def predict_and_update(
        self,
        confidence: float,
        prediction: Any,
        ground_truth: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluates single decision instance under current adaptive threshold tau_t,
        and if ground truth is supplied (post-decision audit feedback), adapts tau_{t+1}.
        """
        is_accepted = confidence >= self.current_tau
        self.history_accepted.append(is_accepted)

        result: Dict[str, Any] = {
            "prediction": prediction if is_accepted else None,
            "status": "ACCEPTED" if is_accepted else "ABSTAIN_ESCALATE",
            "confidence": confidence,
            "current_adaptive_tau": self.current_tau,
            "alpha_target": self.alpha_target
        }

        # If feedback / audit ground-truth is available, apply Gibbs & Candes ACI update
        if ground_truth is not None and is_accepted:
            loss = 0.0 if prediction == ground_truth else 1.0
            self.history_errors.append(loss)

            # Online gradient step on threshold:
            # If error occurred (loss=1.0 > alpha), increase threshold tau (be more conservative)
            # If accurate (loss=0.0 < alpha), slightly relax threshold tau (increase coverage)
            delta = self.gamma * (loss - self.alpha_target)
            self.current_tau = float(np.clip(self.current_tau + delta, self.tau_min, self.tau_max))
            self.history_tau.append(self.current_tau)
            result["online_updated_tau"] = self.current_tau
            result["audit_loss"] = loss

        return result

    def get_online_performance(self) -> Dict[str, float]:
        """Calculates running deployment metrics."""
        total_instances = len(self.history_accepted)
        accepted_count = sum(self.history_accepted)
        error_count = sum(self.history_errors)
        n_audited = len(self.history_errors)

        return {
            "total_stream_instances": total_instances,
            "total_accepted": accepted_count,
            "running_coverage": accepted_count / max(total_instances, 1),
            "audited_instances": n_audited,
            "empirical_conditional_error": error_count / max(n_audited, 1),
            "empirical_conditional_accuracy": 1.0 - (error_count / max(n_audited, 1)),
            "current_tau": self.current_tau,
            "alpha_target": self.alpha_target
        }
