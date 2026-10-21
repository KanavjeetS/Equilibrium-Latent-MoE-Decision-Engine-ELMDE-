# Equilibrium Latent-MoE Decision Engine (ELMDE)
### Overcoming Autoregressive Constrained Decoding Distortion

This repository contains the complete research report, theoretical derivations, visual artifacts, and a modular **production-grade PyTorch implementation and serving microservice** for the **Equilibrium Latent-MoE Decision Engine (ELMDE)**, extracted from the Gemini Deep Research study (*"Jev vs Laya Decision Models"* / *Advanced Non-Aggressive Decision Model*).

---

## 📂 Repository Structure

```
d:\Advanced Non-Aggressive Decision Model\
├── elmde/                               # Production Python Package
│   ├── __init__.py                      # Package entry point
│   ├── config.py                        # Hardware-aligned ELMDEConfig (Tensor Core aligned)
│   ├── nn/                              # Neural Network Primitives
│   │   ├── rms_norm.py                  # De-biased Pre-RMSNorm
│   │   ├── swiglu.py                    # Hardware-aligned SwiGLU FFN
│   │   ├── rope.py                      # Decoupled RoPE (isolated to rotary subspace)
│   │   ├── mla.py                       # Multi-Head Latent Attention with Absorb Mode
│   │   ├── sparse_moe.py                # Routed Sparse MoE with DeepSeek-style shared expert
│   │   ├── equilibrium_core.py          # Contractive Recurrent-Depth MoE with IFT gradients
│   │   ├── heads.py                     # Tri-Modal Constraint Heads (Choice, CORN, Beta)
│   │   └── model.py                     # Unified ELMDEModel Architecture
│   ├── training/                        # Training & Optimization
│   │   ├── losses.py                    # CORN conditional loss & Strictly Proper Scoring Rules
│   │   └── rlcr.py                      # RL with Calibration Rewards under Dr. GRPO
│   ├── conformal/                       # Conformal Prediction
│   │   └── scrc.py                      # Two-Stage Selective Conformal Risk Control
│   └── serving/                         # Production Microservice
│       ├── schemas.py                   # Pydantic v2 I/O validation schemas
│       └── api.py                       # High-performance FastAPI serving application
│
├── demo_inference.py                    # Interactive live demonstration CLI
├── verify_production.py                 # 8-part unit and integration verification suite
├── elmde_model.py                       # Self-contained NumPy reference implementation
├── elmde_architecture.html              # Standalone interactive SVG/HTML architectural diagram
├── myopic_token_masking.html            # Interactive visualizer (Myopic Masking vs Lookahead Alignment)
├── research_report.md                   # Full 95KB Deep Research report & mathematical proofs
└── full_gemini_payload.json             # Raw WIZ RPC conversation payload
```

---

## 🔬 Core Architectural Pillars

1. **Hardware-Aligned Unpadded Bidirectional Backbone**:
   - Zero-padding compute waste via unpadded jagged sequences ($X_{\text{unpad}} \in \mathbb{R}^{N_{\text{total}} \times d}$).
   - Alternating local sliding-window attention ($w = 128, \beta_{\text{local}} = 10,000$) and global attention ($\beta_{\text{global}} = 160,000$).
   - De-biased linear projections and dimensions aligned to multiples of 64/128 for peak Tensor Core tiling.

2. **Multi-Head Latent Attention (MLA) with Commutative Matrix Absorption**:
   - Compresses Key-Value representations into a low-rank latent subspace $c_t^{KV} \in \mathbb{R}^{d_{kv}}$ with Decoupled RoPE $k_t^R \in \mathbb{R}^{d_r}$.
   - **$92.97\%$ reduction in KV cache memory footprint**.
   - **Absorb Mode**: Computes attention directly against the compressed latent without decompressing $K^C$ or $V^C$, matching explicit decompression down to machine epsilon ($8.94 \times 10^{-8}$ deviation in float32).

3. **Contractive Recurrent-Depth Sparse MoE Core**:
   - Overcomes the dense looping scaling penalty ($r^{0.46}$) by routing working latent states $z^{(r)} \in \mathbb{R}^{M \times d}$ through specialized SwiGLU experts across recurrence ticks.
   - Formulated as a damped contractive mapping with Lipschitz constant $\kappa < 1$, guaranteeing geometric convergence to a unique equilibrium fixed point $z^\star$ by the Banach Fixed-Point Theorem.
   - Enables $O(1)$-memory implicit gradients via the **Implicit Function Theorem (IFT)** and state-dependent adaptive halting.

4. **Tri-Modal Constraint-Manifold Projection Heads**:
   - **Heteroscedastic Global Choice Head**: Direct single-pass normalization over valid action candidates with learned temperature $\tau_\phi(z^\star)$, eliminating prefix-sharing distortion ($D_{\text{KL}} = 0$).
   - **Conditional Ordinal Regression (CORN) Head**: Predicts conditional exceedance probabilities $q_j = P(Y > j \mid Y > j-1)$, guaranteeing exact rank monotonicity without parameter tying.
   - **Continuous Beta Density Head**: Parameterizes bounded continuous distributions with analytical epistemic and aleatoric uncertainty.

5. **Strictly Proper Scoring Rules & RLCR**:
   - Group Relative Policy Optimization (GRPO / Dr. GRPO) driven by composite rewards including Logarithmic, Brier, Spherical, and Ranked Probability Scores (RPS).
   - Minimizes Expected Calibration Error (ECE) and prevents mode collapse.

6. **Two-Stage Selective Conformal Risk Control (SCRC)**:
   - Provides finite-sample, distribution-free mathematical error guarantees.
   - Stage 1 gates predictions based on calibrated confidence thresholds; Stage 2 guarantees conditional error bounds ($\le \alpha_{\text{target}} = 0.01$, $\ge 99.0\%$ certified accuracy).

---

## ⚡ Quick Start

### 1. Run the Interactive Demo
```powershell
.venv\Scripts\python.exe demo_inference.py
```

### 2. Run the Comprehensive Verification Suite
```powershell
.venv\Scripts\python.exe verify_production.py
```

### 3. Launch the Production FastAPI Serving API
```powershell
.venv\Scripts\python.exe -m uvicorn elmde.serving:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at `http://localhost:8000/docs`.
