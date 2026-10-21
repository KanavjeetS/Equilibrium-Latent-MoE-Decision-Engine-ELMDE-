"""
Python Client SDK for the Equilibrium Latent-MoE Decision Engine (ELMDE).
Provides clean, synchronous and asynchronous client interfaces for production integration.
"""

from typing import Any, Dict, List, Optional
import httpx


class ELMDEClient:
    """Production client SDK for communicating with the ELMDE Serving API."""
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Checks serving microservice health status."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    def predict_choice(
        self,
        input_ids: List[int],
        candidate_embeddings: List[List[float]],
        valid_mask: Optional[List[bool]] = None
    ) -> Dict[str, Any]:
        """Queries discrete choice prediction head."""
        payload = {
            "input_ids": input_ids,
            "candidate_embeddings": candidate_embeddings,
            "valid_mask": valid_mask
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/v1/predict/choice", json=payload)
            resp.raise_for_status()
            return resp.json()

    def predict_ordinal(self, input_ids: List[int]) -> Dict[str, Any]:
        """Queries rank-consistent CORN ordinal regression head."""
        payload = {"input_ids": input_ids}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/v1/predict/ordinal", json=payload)
            resp.raise_for_status()
            return resp.json()

    def predict_density(self, input_ids: List[int]) -> Dict[str, Any]:
        """Queries continuous bounded Beta density head."""
        payload = {"input_ids": input_ids}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/v1/predict/density", json=payload)
            resp.raise_for_status()
            return resp.json()

    def calibrate_conformal_gate(
        self,
        confidences: List[float],
        losses: List[float],
        alpha_target: float = 0.01
    ) -> Dict[str, Any]:
        """Calibrates the SCRC conformal gate threshold."""
        payload = {
            "confidences": confidences,
            "losses": losses,
            "alpha_target": alpha_target
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/v1/conformal/calibrate", json=payload)
            resp.raise_for_status()
            return resp.json()
