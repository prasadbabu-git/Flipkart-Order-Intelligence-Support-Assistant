from __future__ import annotations
import json

def detect_intent(text: str):
    t=text.lower()
    if any(x in t for x in ['image','photo','picture','what category','classify this']): return 'PRODUCT_CATEGORY'
    if any(x in t for x in ['risk','return risk','will this be returned','probability of return']): return 'RETURN_RISK'
    return 'POLICY'

def few_shot_examples():
    return [
        {'user':'What is the return period for shoes?','intent':'POLICY'},
        {'user':'Can you identify this product image?','intent':'PRODUCT_CATEGORY'},
    ]

def generate(intent, context):
    if intent=='POLICY':
        if not context.get('grounded',False):
            score = float(context.get('score', 0))
            threshold = float(context.get('threshold', 0))
            anchors = context.get('anchors', [])
            return {
                'answer': (
                    f"I could not find a sufficiently grounded policy answer. "
                    f"Retrieval score: {score:.3f}; "
                    f"semantic threshold: {threshold:.3f}; "
                    f"required domain anchor: {bool(anchors)}."
                ),
                'source': 'policy_kb',
                'confidence': round(score, 4),
            }
        hit=context['hits'][0]
        return {'answer':hit['text'],'source':'policy_kb','confidence':round(float(hit['score']),4)}
    if intent == "RETURN_RISK":
        r = context["risk"]

        explanation = r.get("explanation", [])

    risk_factors = [
        item
        for item in explanation
        if item["direction"] == "increases risk"
    ]

    protective_factors = [
        item
        for item in explanation
        if item["direction"] == "reduces risk"
    ]

    lines = [
    (
        f"The estimated return probability is "
        f"{r['return_probability']:.1%}. "
        f"Under the model's calibrated thresholds, "
        f"this falls into the {r['risk_bucket']} risk bucket."
    )
]

    if risk_factors:
        lines.append(
    "Model-derived factors associated with higher predicted return risk:"
)
        for item in risk_factors[:3]:
            lines.append(
                f"• {item['description']}"
            )

    if protective_factors:
        lines.append(
    "Model-derived factors associated with lower predicted return risk:"
)
        for item in protective_factors[:2]:
            lines.append(
                f"• {item['description']}"
            )

    return {
        "answer": "\n".join(lines),
        "source": "return_risk_tool",
        "confidence": round(
            float(r["return_probability"]),
            4
        ),
    }