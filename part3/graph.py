from __future__ import annotations
from pathlib import Path
from typing import TypedDict, Optional

from part3.mock_llm import detect_intent, generate, few_shot_examples
from part3.guardrails import check_input
from part3.rag import LocalRetriever
from part3.tools import check_return_risk, classify_product_image

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE=True
except Exception:
    LANGGRAPH_AVAILABLE=False

class State(TypedDict, total=False):
    user_text:str
    intent:str
    blocked:bool
    grounded:bool
    score:float
    threshold:float
    hits:list
    risk:dict
    classification:dict
    answer:dict
    conversation:dict

retriever=LocalRetriever()

class SupportGraph:
    def __init__(self):
        self.graph=self._build() if LANGGRAPH_AVAILABLE else None
    def intent_node(self,s):
        s['intent']=detect_intent(s['user_text']); return s
    def policy_node(self,s):
        g=retriever.grounded(s['user_text'],threshold=0.60)
        s.update(g); return s
    def risk_node(self,s):
        # Demo state may contain an order feature dict encoded by caller.
        order=s.get('conversation',{}).get('order_features')
        if not order: raise ValueError('RETURN_RISK requires conversation[order_features].')
        s['risk']=check_return_risk(order); return s
    def image_node(self,s):
        path=s.get('conversation',{}).get('image_path')
        if not path: raise ValueError('PRODUCT_CATEGORY requires conversation[image_path].')
        s['classification']=classify_product_image(path); return s
    def response_node(self,s):
        s['answer']=generate(s['intent'],s); return s
    def _build(self):
        g=StateGraph(State)
        g.add_node('intent',self.intent_node); g.add_node('policy_rag',self.policy_node)
        g.add_node('risk_tool',self.risk_node); g.add_node('image_tool',self.image_node)
        g.add_node('response',self.response_node)
        g.add_edge(START,'intent')
        g.add_conditional_edges('intent',lambda s:s['intent'],{'POLICY':'policy_rag','RETURN_RISK':'risk_tool','PRODUCT_CATEGORY':'image_tool'})
        g.add_edge('policy_rag','response'); g.add_edge('risk_tool','response'); g.add_edge('image_tool','response'); g.add_edge('response',END)
        return g.compile()
    def invoke(self,user_text,conversation=None):
        conversation=conversation or {}
        guard=check_input(user_text)
        if guard['blocked']:
            return {'answer':{'answer':'I cannot follow a request that attempts to override the support assistant instructions.','source':'guardrail','confidence':1.0},'blocked':True,'intent':'POLICY'}
        state={'user_text':user_text,'conversation':conversation}
        if self.graph is not None:
            return self.graph.invoke(state)
        s=self.intent_node(state)
        if s['intent']=='POLICY': s=self.policy_node(s)
        elif s['intent']=='RETURN_RISK': s=self.risk_node(s)
        else: s=self.image_node(s)
        return self.response_node(s)

def create_app(): return SupportGraph()
