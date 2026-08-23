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
            return {'answer':f"I could not find a sufficiently grounded policy answer. Retrieval score {context.get('score',0):.3f} is below the threshold {context.get('threshold',0):.3f}.",'source':'policy_kb','confidence':round(float(context.get('score',0)),4)}
        hit=context['hits'][0]
        return {'answer':hit['text'],'source':'policy_kb','confidence':round(float(hit['score']),4)}
    if intent=='RETURN_RISK':
        r=context['risk']
        return {'answer':f"The estimated return probability is {r['return_probability']:.1%}, giving a {r['risk_bucket']} return-risk rating.",'source':'return_risk_tool','confidence':round(r['return_probability'],4)}
    c=context['classification']
    return {'answer':f"The product is classified as {c['label']} with {c['confidence']:.1%} confidence.",'source':'image_classifier_tool','confidence':round(float(c['confidence']),4)}
