from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from .order_engine import OrderEngine

INJECTION = re.compile(r"ignore (all|any|the|previous)|system prompt|developer message|jailbreak|pretend you are", re.I)

@dataclass
class Session:
    order_id: int | None = None
    history: list[dict] = field(default_factory=list)

class SupportAgent:
    def __init__(self):
        self.orders = OrderEngine()
        self.session = Session()

    def intent(self, text: str) -> str:
        t = text.lower()
        if any(x in t for x in ["return risk", "risk", "likely to return"]): return "RETURN_RISK"
        if any(x in t for x in ["image", "photo", "picture", "what category"]): return "PRODUCT_CATEGORY"
        if any(x in t for x in ["where is", "track", "status", "order"]): return "ORDER"
        return "POLICY"

    def chat(self, text: str, order_id: int | None = None) -> dict:
        if INJECTION.search(text):
            return {"answer": "I can help with order support, but I can't follow instructions that attempt to override my operating rules.", "source": "guardrail", "confidence": 1.0}
        if order_id is not None:
            self.session.order_id = int(order_id)
        m = re.search(r"\b(?:order|#)\s*([0-9]{1,6})\b", text, re.I)
        if m: self.session.order_id = int(m.group(1))
        intent = self.intent(text)
        try:
            if intent in {"ORDER", "RETURN_RISK"}:
                if self.session.order_id is None:
                    return {"answer": "Please provide the order ID so I can check it.", "source": "order_engine", "confidence": 0.98}
                result = self.orders.order_summary(self.session.order_id)
                if intent == "RETURN_RISK":
                    r = result["risk"]
                    answer = f"Order {self.session.order_id} has {r['risk'].lower()} return risk with probability {r['probability']:.1%}."
                    source = "return_risk_tool"
                else:
                    o = result["order"]
                    answer = f"Order {self.session.order_id} is for {o['product_category']} at ₹{o['price_inr']:.0f}, paid via {o['payment_method']}."
                    source = "order_lookup_tool"
                return {"answer": answer, "source": source, "confidence": 0.95}
            return {"answer": "I can answer return, refund, delivery, exchange and related support-policy questions using the project knowledge base.", "source": "policy_kb", "confidence": 0.75}
        except KeyError as exc:
            return {"answer": str(exc).strip("'"), "source": "order_engine", "confidence": 1.0}
