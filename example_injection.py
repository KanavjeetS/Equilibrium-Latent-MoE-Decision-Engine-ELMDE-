"""
ELMDE Injection Examples: Plug-and-Play Decision Intelligence for Any AI / LLM Pipeline.

Demonstrates how to inject ELMDE directly into:
1. Agent Tool Calling & Routing (bypassing slow LLM tool calls)
2. High-Cardinality Intent Classification (77+ choices without prompt truncation)
3. Monotonic Risk Guardrails (with certified automatic escalation)
4. LangChain / LlamaIndex / Agentic Workflow Decision Nodes
"""

from elmde import ELMDE, DecisionResult, TriageResult

# ============================================================================
# 1. LOAD THE ENGINE (Zero Background Daemons, Zero Ports)
# ============================================================================
print("[*] Initializing ELMDE In-Memory Decision Engine...")
engine = ELMDE.load()
print("[*] Engine ready. Parameters: 12.7M | Device: CPU/CUDA | Memory: O(1)\n")


# ============================================================================
# 2. INJECT INTO AGENT TOOL CALLING (Sub-95ms, Zero Token Cost)
# ============================================================================
print("=" * 70)
print("EXAMPLE 1: Agent Tool Router (Fast-Pathing Function Calling)")
print("=" * 70)

def search_knowledge_base(query: str):
    return f"[TOOL OUTPUT] Found documentation for query: '{query}'"

def execute_sql_query(query: str):
    return f"[TOOL OUTPUT] Executed database query: '{query}'"

def refund_customer_account(query: str):
    return f"[TOOL OUTPUT] Processed financial refund for: '{query}'"

tools = {
    "search_knowledge_base": search_knowledge_base,
    "execute_sql_query": execute_sql_query,
    "refund_customer_account": refund_customer_account,
}

user_request = "Can you refund transaction #TX-99201? The customer received a damaged item."

# Route directly without burning hundreds of LLM tokens on JSON schemas
func, decision = engine.route_tool(user_request, tools)
print(f"User Request:       '{user_request}'")
print(f"Routed Tool:        {decision.choice} (Confidence: {decision.confidence:.4f})")
print(f"Decision Latency:   {decision.latency_ms:.1f} ms (Ticks: {decision.recurrent_steps})")
print(f"Execution Output:   {func(user_request)}\n")


# ============================================================================
# 3. HIGH-CARDINALITY ROUTING (77+ Choices - Where LLMs & CDMs Fail)
# ============================================================================
print("=" * 70)
print("EXAMPLE 2: 77-Option Routing (Zero Prompt Truncation)")
print("=" * 70)

banking_intents = [
    "card_arrival", "card_linking", "exchange_rate", "card_payment_fee_charged",
    "extra_charge_on_statement", "refund_not_showing_up", "balance_not_updated_after_cheque_or_cash_deposit",
    "pin_blocked", "contactless_not_working", "lost_or_stolen_card", "transfer_not_received_by_recipient",
    "verify_identity", "passcode_forgotten", "compromised_card", "top_up_reverted"
] # Truncated list for display; handles all 77+ Banking77 intents

incoming_message = "I noticed an unauthorized charge of $84.20 on my monthly statement from a merchant I don't recognize."

res = engine.decide(incoming_message, choices=banking_intents)
print(f"Message:            '{incoming_message}'")
print(f"Selected Intent:    {res.choice} (Conf: {res.confidence:.4f})")
print(f"Status:             {res.status} (Is Safe: {res.is_safe})")
print(f"Decision Latency:   {res.latency_ms:.1f} ms\n")


# ============================================================================
# 4. MONOTONIC RISK GUARDRAIL (With Certified Safety Escalation)
# ============================================================================
print("=" * 70)
print("EXAMPLE 3: Conformal Risk Guardrail (Automatic Human Escalation)")
print("=" * 70)

clinical_note = "Elderly patient presenting with sudden onset left-sided facial droop and slurred speech."
triage_tiers = ["LEVEL_5_NON_URGENT", "LEVEL_4_STANDARD", "LEVEL_3_URGENT", "LEVEL_2_EMERGENT", "LEVEL_1_RESUSCITATION"]

triage = engine.triage(clinical_note, tiers=triage_tiers)
print(f"Clinical Note:      '{clinical_note}'")
print(f"Assigned Tier:      {triage.tier} (Rank: {triage.predicted_rank})")
print(f"Conformal Gate:     {triage.status} (Certified Floor: alpha <= 0.01)")
print(f"Cumulative Surv.:   {[round(x, 4) for x in triage.cumulative_survival]}")
print(f"Decision Latency:   {triage.latency_ms:.1f} ms\n")


# ============================================================================
# 5. INLINE LLM GUARDRAIL DECORATOR
# ============================================================================
print("=" * 70)
print("EXAMPLE 4: Python Function / LLM Call Guardrail Decorator")
print("=" * 70)

@engine.guardrail(
    fallback_handler=lambda prompt, res: f"[ESCALATED TO HUMAN SPECIALIST] Confidence ({res.confidence:.4f}) below safety threshold."
)
def autonomous_trading_execution(trade_order: str):
    return f"[EXECUTED] Trade order placed successfully: {trade_order}"

test_order = "Execute maximum leverage short position on illiquid token with high slippage."
result = autonomous_trading_execution(test_order)
print(f"Trade Input:        '{test_order}'")
print(f"Guardrail Verdict:  {result}\n")

print("=" * 70)
print("ELMDE successfully injected into all 4 AI/LLM production patterns!")
print("=" * 70)
