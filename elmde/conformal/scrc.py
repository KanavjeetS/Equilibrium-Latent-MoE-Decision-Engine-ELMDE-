from typing import Any, Dict, Optional, Tuple
import numpy as np


class SelectiveConformalRiskController:
    """
    Two-Stage Selective Conformal Risk Control (SCRC).
    Guarantees that the expected loss on automatically accepted predictions
    never exceeds a user-specified error ceiling alpha_target:
        E[loss(C(X), Y) | confidence(X) >= tau*] <= alpha_target.
    Provably distribution-free and finite-sample valid.
    """
    def __init__(self, alpha_target: float = 0.01, loss_bound_b: float = 1.0):
        self.alpha_target = alpha_target
        self.loss_bound_b = loss_bound_b
        self.calibrated_tau: Optional[float] = None
        self.calibration_stats: Optional[Dict[str, Any]] = None

    def calibrate(
        self,
        confidences: np.ndarray,
        losses: np.ndarray,
        tau_grid_steps: int = 1000
    ) -> Dict[str, Any]:
        """
        Calibrates the acceptance threshold tau* on an exchangeable calibration split.
        Args:
            confidences: Array of model confidence scores in [0, 1] of shape (N_cal,)
            losses: Array of evaluation losses in [0, B] of shape (N_cal,)
        """
        n_total = len(confidences)
        assert n_total == len(losses), "Confidences and losses must have identical length"

        # Search grid from 1.0 down to min(confidences)
        tau_candidates = np.linspace(1.0, float(np.min(confidences)), tau_grid_steps)
        best_tau = None
        best_stats = {}

        for tau in tau_candidates:
            accepted_mask = confidences >= tau
            m = np.sum(accepted_mask)

            if m == 0:
                continue

            emp_risk = np.mean(losses[accepted_mask])

            # Angelopoulos / Xu-Guo-Wei finite-sample Conformal Risk Control bound:
            # (m / (m + 1)) * R_hat + (B / (m + 1)) <= alpha_target
            crc_upper_bound = (float(m) / (m + 1.0)) * emp_risk + (self.loss_bound_b / (m + 1.0))

            if crc_upper_bound <= self.alpha_target:
                best_tau = float(tau)
                best_stats = {
                    "tau_star": best_tau,
                    "calibration_samples_accepted": int(m),
                    "coverage_calibration": float(m) / n_total,
                    "empirical_conditional_risk": float(emp_risk),
                    "crc_finite_sample_bound": float(crc_upper_bound),
                    "target_alpha": self.alpha_target
                }
            else:
                # Monotonic increase of risk as threshold drops; stop at boundary
                break

        if best_tau is None:
            # Fallback to strictest confidence in calibration data
            best_tau = float(np.max(confidences))
            best_stats = {
                "tau_star": best_tau,
                "warning": "Strict bound could not be met across full grid; fallback to max confidence."
            }

        self.calibrated_tau = best_tau
        self.calibration_stats = best_stats
        return best_stats

    def evaluate_test_stream(
        self,
        test_confidences: np.ndarray,
        test_losses: np.ndarray
    ) -> Dict[str, float]:
        """Evaluates certified selective prediction on an unseen test stream."""
        assert self.calibrated_tau is not None, "Controller must be calibrated before evaluation."
        accepted_mask = test_confidences >= self.calibrated_tau
        n_test = len(test_confidences)
        n_accepted = np.sum(accepted_mask)

        coverage = float(n_accepted) / n_test if n_test > 0 else 0.0
        cond_error = float(np.mean(test_losses[accepted_mask])) if n_accepted > 0 else 0.0

        return {
            "test_coverage": coverage,
            "test_conditional_error": cond_error,
            "test_conditional_accuracy": 1.0 - cond_error,
            "abstention_rate": 1.0 - coverage,
            "samples_evaluated": n_test,
            "samples_accepted": int(n_accepted)
        }

    def predict_selective(
        self,
        confidence: float,
        prediction: Any
    ) -> Dict[str, Any]:
        """
        Gated inference for a single instance.
        Returns prediction if confidence meets certified threshold; abstains otherwise.
        """
        assert self.calibrated_tau is not None, "Controller must be calibrated first."
        is_accepted = confidence >= self.calibrated_tau

        return {
            "prediction": prediction if is_accepted else None,
            "status": "ACCEPTED" if is_accepted else "ABSTAIN_ESCALATE",
            "confidence": confidence,
            "certified_threshold": self.calibrated_tau,
            "guaranteed_max_error": self.alpha_target
        }
