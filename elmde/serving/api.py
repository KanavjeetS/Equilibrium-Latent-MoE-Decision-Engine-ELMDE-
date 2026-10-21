from contextlib import asynccontextmanager
from typing import Dict, Optional
import os
import json
import time
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from ..config import ELMDEConfig
from ..nn.model import ELMDEModel
from ..conformal.scrc import SelectiveConformalRiskController
from .schemas import (
    ChoicePredictionRequest,
    ChoicePredictionResponse,
    OrdinalPredictionRequest,
    OrdinalPredictionResponse,
    DensityPredictionRequest,
    DensityPredictionResponse,
    ConformalCalibrationRequest,
    ConformalCalibrationResponse,
)

# Prometheus Metrics
REQUESTS_TOTAL = Counter("elmde_requests_total", "Total requests processed", ["endpoint", "status"])
INFERENCE_LATENCY = Histogram("elmde_inference_latency_seconds", "Inference latency in seconds", ["endpoint"])
DECISIONS_ACCEPTED = Counter("elmde_decisions_accepted_total", "Total decisions automatically accepted by SCRC gate")
DECISIONS_ESCALATED = Counter("elmde_decisions_escalated_total", "Total decisions escalated to human review by SCRC gate")

# Global model and controller references
_model: Optional[ELMDEModel] = None
_controller: Optional[SelectiveConformalRiskController] = None
_tokenizer = None


def _resolve_input_ids(request, device: torch.device, vocab_size: int = 30522) -> torch.Tensor:
    if request.input_ids is not None:
        ids = [min(int(x), vocab_size - 1) for x in request.input_ids]
        if len(ids) == 0:
            ids = [101]
        return torch.tensor([ids], dtype=torch.long, device=device)
    elif request.text is not None:
        if _tokenizer is not None:
            try:
                tensor_ids = _tokenizer.encode_text(request.text, max_length=128, return_tensors="pt")
                clamped = torch.clamp(tensor_ids, 0, vocab_size - 1)
                return clamped.to(device)
            except Exception:
                ids = [ord(c) % vocab_size for c in request.text[:128]]
                return torch.tensor([ids], dtype=torch.long, device=device)
        else:
            ids = [ord(c) % vocab_size for c in request.text[:128]]
            return torch.tensor([ids], dtype=torch.long, device=device)
    else:
        raise HTTPException(status_code=400, detail="Either 'text' or 'input_ids' must be provided in request payload.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _controller, _tokenizer

    checkpoint_path = os.path.join("checkpoints", "elmde_production.pt")
    cal_path = os.path.join("checkpoints", "scrc_calibration.json")

    # 1. Load trained checkpoint if exists, otherwise initialize clean model
    if os.path.exists(checkpoint_path):
        print(f"[*] Loading production checkpoint from: {checkpoint_path}")
        _model = ELMDEModel.from_pretrained(checkpoint_path)
    else:
        print("[*] No trained checkpoint found; initializing baseline architecture")
        config = ELMDEConfig()
        _model = ELMDEModel(config)

    _model.eval()

    # 2. Load conformal calibration if exists
    _controller = SelectiveConformalRiskController(alpha_target=_model.config.scrc_alpha_target)
    if os.path.exists(cal_path):
        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
                _controller.calibrated_tau = float(cal_data.get("tau_star", 0.85))
                _controller.calibration_stats = cal_data
                print(f"[*] Restored calibrated SCRC threshold: {_controller.calibrated_tau:.4f}")
        except Exception:
            _controller.calibrated_tau = 0.85
    else:
        _controller.calibrated_tau = 0.85

    # 3. Initialize tokenizer for raw text serving matching model vocab_size
    try:
        from ..data.tokenizer import ELMDETokenizer
        model_name = "bert-base-uncased" if _model.config.vocab_size == 30522 else "answerdotai/ModernBERT-base"
        _tokenizer = ELMDETokenizer(model_name_or_path=model_name)
        print(f"[*] Serving tokenizer: {model_name} (vocab: {_model.config.vocab_size}) initialized.")
    except Exception as e:
        print(f"[!] Tokenizer initialization warning: {e}")
        _tokenizer = None

    yield


app = FastAPI(
    title="Equilibrium Latent-MoE Decision Engine (ELMDE) API",
    description="Production-grade serving API for non-aggressive structured decision models with exact constraint manifolds and conformal risk guarantees.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def get_dashboard():
    """Serves the interactive ELMDE Decision Console."""
    dashboard_path = os.path.join(os.getcwd(), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ELMDE API Running</h1><p>Visit <a href='/docs'>/docs</a> for OpenAPI spec.</p>"


@app.get("/architecture", response_class=HTMLResponse, tags=["UI"])
def get_architecture_showcase():
    """Serves the interactive animated architecture diagram showcase."""
    arch_path = os.path.join(os.getcwd(), "animated_architecture.html")
    if os.path.exists(arch_path):
        with open(arch_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Architecture Showcase</h1><p>animated_architecture.html not found.</p>"


@app.get("/metrics", tags=["Observability"])
def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "HEALTHY",
        "model_loaded": _model is not None,
        "conformal_controller_ready": _controller is not None and _controller.calibrated_tau is not None,
        "certified_threshold_tau": _controller.calibrated_tau if _controller else None,
        "device": "cpu" if _model is None else next(_model.parameters()).device.type,
        "parameters": sum(p.numel() for p in _model.parameters()) if _model else 0
    }


@app.post(
    "/v1/predict/choice",
    response_model=ChoicePredictionResponse,
    tags=["Inference"]
)
def predict_choice(request: ChoicePredictionRequest):
    if _model is None or _controller is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    start_time = time.perf_counter()
    device = next(_model.parameters()).device
    input_ids = _resolve_input_ids(request, device, _model.config.vocab_size)
    cand_tensor = torch.tensor([request.candidate_embeddings], dtype=torch.float32, device=device)
    mask_tensor = torch.tensor([request.valid_mask], dtype=torch.bool, device=device) if request.valid_mask else None

    with torch.no_grad():
        probs, tau, metrics = _model.forward_choice(input_ids, cand_tensor, mask_tensor)

    prob_list = probs.squeeze(0).cpu().tolist()
    top_idx = int(np.argmax(prob_list))
    top_conf = float(prob_list[top_idx])
    decision = _controller.predict_selective(top_conf, top_idx)

    # Telemetry
    elapsed = time.perf_counter() - start_time
    INFERENCE_LATENCY.labels(endpoint="choice").observe(elapsed)
    REQUESTS_TOTAL.labels(endpoint="choice", status="success").inc()
    if decision["status"] == "ACCEPTED":
        DECISIONS_ACCEPTED.inc()
    else:
        DECISIONS_ESCALATED.inc()

    return ChoicePredictionResponse(
        probabilities=prob_list,
        selected_index=top_idx,
        temperature=float(tau.item()),
        status=decision["status"],
        confidence=top_conf,
        recurrent_steps=int(metrics["steps_taken"].item()),
        equilibrium_residual=float(metrics["final_residual"].item())
    )


@app.post(
    "/v1/predict/ordinal",
    response_model=OrdinalPredictionResponse,
    tags=["Inference"]
)
def predict_ordinal(request: OrdinalPredictionRequest):
    if _model is None or _controller is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    start_time = time.perf_counter()
    device = next(_model.parameters()).device
    input_ids = _resolve_input_ids(request, device, _model.config.vocab_size)

    with torch.no_grad():
        class_probs, cum_surv, metrics = _model.forward_ordinal(input_ids)

    probs_list = class_probs.squeeze(0).cpu().tolist()
    surv_list = cum_surv.squeeze(0).cpu().tolist()
    top_rank = int(np.argmax(probs_list))
    top_conf = float(probs_list[top_rank])
    decision = _controller.predict_selective(top_conf, top_rank)

    # Telemetry
    elapsed = time.perf_counter() - start_time
    INFERENCE_LATENCY.labels(endpoint="ordinal").observe(elapsed)
    REQUESTS_TOTAL.labels(endpoint="ordinal", status="success").inc()
    if decision["status"] == "ACCEPTED":
        DECISIONS_ACCEPTED.inc()
    else:
        DECISIONS_ESCALATED.inc()

    return OrdinalPredictionResponse(
        class_probabilities=probs_list,
        predicted_rank=top_rank,
        cumulative_survival=surv_list,
        confidence=top_conf,
        status=decision["status"],
        recurrent_steps=int(metrics["steps_taken"].item()),
        equilibrium_residual=float(metrics["final_residual"].item())
    )


@app.post(
    "/v1/predict/density",
    response_model=DensityPredictionResponse,
    tags=["Inference"]
)
def predict_density(request: DensityPredictionRequest):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    start_time = time.perf_counter()
    device = next(_model.parameters()).device
    input_ids = _resolve_input_ids(request, device, _model.config.vocab_size)

    with torch.no_grad():
        mean, var, metrics = _model.forward_density(input_ids)

    m_val = float(mean.item())
    v_val = float(var.item())
    alpha_val = float(m_val * (m_val * (1.0 - m_val) / max(v_val, 1e-7) - 1.0))
    beta_val = float((1.0 - m_val) * (m_val * (1.0 - m_val) / max(v_val, 1e-7) - 1.0))

    elapsed = time.perf_counter() - start_time
    INFERENCE_LATENCY.labels(endpoint="density").observe(elapsed)
    REQUESTS_TOTAL.labels(endpoint="density", status="success").inc()

    return DensityPredictionResponse(
        expected_value=m_val,
        uncertainty_variance=v_val,
        alpha_param=max(alpha_val, 0.01),
        beta_param=max(beta_val, 0.01),
        recurrent_steps=int(metrics["steps_taken"].item())
    )


@app.post(
    "/v1/conformal/calibrate",
    response_model=ConformalCalibrationResponse,
    tags=["Risk Control"]
)
def calibrate_conformal_gate(request: ConformalCalibrationRequest):
    global _controller
    if _controller is None:
        _controller = SelectiveConformalRiskController(alpha_target=request.alpha_target or 0.01)
    else:
        _controller.alpha_target = request.alpha_target or 0.01

    conf_arr = np.array(request.confidences, dtype=np.float64)
    loss_arr = np.array(request.losses, dtype=np.float64)

    stats = _controller.calibrate(conf_arr, loss_arr)

    # Persist calibration state to disk
    os.makedirs("checkpoints", exist_ok=True)
    cal_path = os.path.join("checkpoints", "scrc_calibration.json")
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return ConformalCalibrationResponse(
        tau_star=stats["tau_star"],
        target_alpha=_controller.alpha_target,
        calibration_samples_accepted=stats.get("calibration_samples_accepted", len(conf_arr)),
        coverage_calibration=stats.get("coverage_calibration", 1.0),
        empirical_conditional_risk=stats.get("empirical_conditional_risk", 0.0)
    )
