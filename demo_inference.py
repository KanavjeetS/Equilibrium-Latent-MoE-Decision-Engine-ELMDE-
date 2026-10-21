"""
Interactive Demonstration of the Equilibrium Latent-MoE Decision Engine (ELMDE).
Demonstrates live discrete choice selection, rank-consistent ordinal evaluation,
continuous density estimation, and selective conformal risk gating.
"""

import sys
import os
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')

from elmde.config import ELMDEConfig
from elmde.nn.model import ELMDEModel
from elmde.conformal.scrc import SelectiveConformalRiskController


def run_demo():
    print("=" * 75)
    print("  EQUILIBRIUM LATENT-MoE DECISION ENGINE (ELMDE) - LIVE DEMO")
    print("=" * 75)

    # 1. Initialize configuration and model
    cfg = ELMDEConfig(
        vocab_size=50368,
        d_model=768,
        n_heads=12,
        d_head=64,
        d_rope=32,
        d_kv_latent=128,
        d_q_latent=256,
        d_ff=2048,
        n_encoder_layers=4,
        n_latent_slots=8,
        n_experts=8,
        top_k_experts=2,
        max_recurrent_steps=6,
        contraction_alpha=0.45,
        num_ordinal_classes=5,
        scrc_alpha_target=0.01  # 99.0% certified accuracy floor
    )
    print(f"[*] Initialized ELMDE Architecture:")
    print(f"    - Hidden Dimension (d_model): {cfg.d_model} (12 * 64)")
    print(f"    - Multi-Head Latent Attention: {cfg.n_heads} heads, d_kv={cfg.d_kv_latent} (83.3% KV compression)")
    print(f"    - Recurrent Core: {cfg.n_latent_slots} working slots, {cfg.n_experts} experts (Top-{cfg.top_k_experts} routed)")
    print(f"    - Contractive Fixed-Point: alpha={cfg.contraction_alpha}, halt_tol={cfg.halt_tol}")
    print()

    # Check for trained production checkpoint
    checkpoint_path = "checkpoints/elmde_production.pt"
    if os.path.exists(checkpoint_path):
        print(f"[*] Found production checkpoint! Loading from: {checkpoint_path}")
        model = ELMDEModel.from_pretrained(checkpoint_path)
        cfg = model.config
    else:
        model = ELMDEModel(cfg)

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Total Model Parameters: {total_params:,}")
    print()

    # 2. Simulate Input Context
    token_ids = torch.randint(100, 10000, (1, 32))
    print(f"[*] Context Input Stream: 32 tokens ingested")

    # 3. Demonstration A: Heteroscedastic Global Choice Head
    print("\n" + "-" * 75)
    print("  DEMONSTRATION A: DISCRETE CONSTRAINT SATISFACTION (GLOBAL CHOICE)")
    print("-" * 75)
    action_names = [
        "Action 0: Direct Approval (Standard Limit)",
        "Action 1: Conditional Approval (Requires Secondary Collateral)",
        "Action 2: Disqualified (Violates Debt-to-Income Constraint)",
        "Action 3: Escalation to Senior Risk Officer",
        "Action 4: Request Additional Financial Documentation"
    ]
    num_candidates = len(action_names)
    candidate_embeddings = torch.randn(1, num_candidates, cfg.d_model)

    # Hard grammar constraint: Action 2 is disqualified by external policy
    valid_mask = torch.tensor([[True, True, False, True, True]])

    with torch.no_grad():
        probs, tau, metrics = model.forward_choice(token_ids, candidate_embeddings, valid_mask)

    prob_list = probs.squeeze(0).numpy()
    print(f"  Fixed-Point Equilibrium: Reached in {metrics['steps_taken'].item()} steps (Residual: {metrics['final_residual'].item():.4e})")
    print(f"  Learned Heteroscedastic Temperature tau: {tau.item():.4f}")
    print("\n  Candidate Probability Distribution:")
    for name, p in zip(action_names, prob_list):
        bar = "#" * int(p * 35)
        status = "[VALID]" if p > 0.0 else "[MASKED BY CONSTRAINT]"
        print(f"    {name:<62} {p*100:5.2f}% {bar:<35} {status}")

    best_action_idx = int(np.argmax(prob_list))
    print(f"\n  >> Top Choice: {action_names[best_action_idx]} (Probability: {prob_list[best_action_idx]*100:.2f}%)")

    # 4. Demonstration B: CORN Rank-Consistent Ordinal Head
    print("\n" + "-" * 75)
    print("  DEMONSTRATION B: CONDITIONAL ORDINAL REGRESSION (CORN)")
    print("-" * 75)
    risk_tiers = ["Tier 0: Minimal Risk", "Tier 1: Low Risk", "Tier 2: Moderate Risk", "Tier 3: Elevated Risk", "Tier 4: Critical Default Risk"]

    with torch.no_grad():
        class_probs, cum_surv, metrics = model.forward_ordinal(token_ids)

    class_probs_arr = class_probs.squeeze(0).numpy()
    cum_surv_arr = cum_surv.squeeze(0).numpy()

    print(f"  CORN Cumulative Exceedance Probabilities P(Risk > k):")
    for k in range(cfg.num_ordinal_classes - 1):
        print(f"    P(Risk > Tier {k}): {cum_surv_arr[k]*100:5.2f}%")

    print("\n  Resolved Exact Point-Mass Distribution (sums identically to 100%):")
    for tier, p in zip(risk_tiers, class_probs_arr):
        bar = "#" * int(p * 35)
        print(f"    {tier:<32} {p*100:5.2f}% {bar}")

    # Check rank monotonicity
    diffs = np.diff(cum_surv_arr)
    is_monotonic = bool(np.all(diffs <= 0.0))
    print(f"\n  >> Strict Mathematical Rank Monotonicity Guaranteed: {is_monotonic}")

    # 5. Demonstration C: Selective Conformal Risk Control (SCRC)
    print("\n" + "-" * 75)
    print("  DEMONSTRATION C: TWO-STAGE SELECTIVE CONFORMAL RISK CONTROL (SCRC)")
    print("-" * 75)
    controller = SelectiveConformalRiskController(alpha_target=0.01)

    # Simulate exchangeable calibration split
    rng = np.random.default_rng(2026)
    n_cal = 5000
    cal_conf = 0.5 + 0.5 * rng.beta(7.0, 1.5, size=n_cal)
    cal_loss = (rng.random(n_cal) >= cal_conf).astype(np.float64)

    cal_stats = controller.calibrate(cal_conf, cal_loss)
    print(f"  Calibrated Gate on {n_cal:,} exchangeable instances:")
    print(f"    - Target Error Ceiling (alpha_target): {controller.alpha_target*100:.1f}% (Certifies >= 99% accuracy)")
    print(f"    - Certified Decision Threshold tau*:   {cal_stats['tau_star']:.4f}")
    print(f"    - Empirical Calibration Coverage:      {cal_stats.get('coverage_calibration', 1.0)*100:.2f}%")

    # Evaluate current inference instance under the certified gate
    instance_confidence = float(np.max(class_probs_arr))
    decision = controller.predict_selective(instance_confidence, best_action_idx)
    print(f"\n  Live Decision Gating for Current Case:")
    print(f"    - Model Predictive Confidence: {instance_confidence*100:.2f}%")
    print(f"    - Required Gate Threshold:     {cal_stats['tau_star']*100:.2f}%")
    print(f"    - Conformal Certification:     {decision['status']}")
    if decision['status'] == "ACCEPTED":
        print(f"    - Result: AUTOMATICALLY EXECUTED (Certified risk <= 1.0%)")
    else:
        print(f"    - Result: ESCALATED TO HUMAN AUDITOR (Epistemic ambiguity detected)")

    print("\n" + "=" * 75)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
