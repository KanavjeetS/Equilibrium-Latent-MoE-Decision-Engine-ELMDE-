from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChoicePredictionRequest(BaseModel):
    input_ids: List[int] = Field(..., description="Token sequence IDs")
    candidate_embeddings: List[List[float]] = Field(..., description="C candidate embeddings of dimension d_model")
    valid_mask: Optional[List[bool]] = Field(None, description="Optional boolean mask over valid candidates")


class ChoicePredictionResponse(BaseModel):
    probabilities: List[float] = Field(..., description="Exact Bayesian probability over candidates")
    selected_index: int = Field(..., description="Candidate index with highest posterior probability")
    temperature: float = Field(..., description="Learned heteroscedastic temperature tau")
    status: str = Field(..., description="'ACCEPTED' or 'ABSTAIN_ESCALATE'")
    confidence: float = Field(..., description="Top choice probability score")
    recurrent_steps: int = Field(..., description="Number of recurrent MoE steps taken to reach equilibrium")
    equilibrium_residual: float = Field(..., description="Fixed-point residual norm ||z^(r+1) - z^(r)||")


class OrdinalPredictionRequest(BaseModel):
    input_ids: List[int] = Field(..., description="Token sequence IDs")


class OrdinalPredictionResponse(BaseModel):
    class_probabilities: List[float] = Field(..., description="Exact probability distribution over K ordinal classes")
    predicted_rank: int = Field(..., description="Most likely ordinal rank (argmax)")
    cumulative_survival: List[float] = Field(..., description="Strictly monotonic survival probabilities P(Y > k)")
    confidence: float = Field(..., description="Confidence score for conformal gating")
    status: str = Field(..., description="'ACCEPTED' or 'ABSTAIN_ESCALATE'")
    recurrent_steps: int = Field(..., description="Recurrent steps taken to equilibrium")
    equilibrium_residual: float = Field(..., description="Fixed-point residual norm")


class DensityPredictionRequest(BaseModel):
    input_ids: List[int] = Field(..., description="Token sequence IDs")


class DensityPredictionResponse(BaseModel):
    expected_value: float = Field(..., description="Posterior mean in (0, 1)")
    uncertainty_variance: float = Field(..., description="Analytical predictive variance")
    alpha_param: float = Field(..., description="Beta distribution alpha parameter")
    beta_param: float = Field(..., description="Beta distribution beta parameter")
    recurrent_steps: int = Field(..., description="Recurrent steps taken to equilibrium")


class ConformalCalibrationRequest(BaseModel):
    confidences: List[float] = Field(..., description="Validation confidence scores in [0, 1]")
    losses: List[float] = Field(..., description="Validation binary or bounded losses in [0, 1]")
    alpha_target: Optional[float] = Field(0.01, description="Target error ceiling (e.g. 0.01 for 99% accuracy)")


class ConformalCalibrationResponse(BaseModel):
    tau_star: float = Field(..., description="Certified acceptance threshold")
    target_alpha: float = Field(..., description="Certified error ceiling")
    calibration_samples_accepted: int
    coverage_calibration: float
    empirical_conditional_risk: float
