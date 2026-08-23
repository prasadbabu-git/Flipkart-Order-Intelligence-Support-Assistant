import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.agent import SupportAgent
from core.order_engine import OrderEngine

st.set_page_config(page_title="Flipkart Order Intelligence", page_icon="🛍️", layout="wide")
st.title("🛍️ Flipkart Order Intelligence & Support Assistant")
st.caption("ML return-risk intelligence • Computer vision • RAG • LangGraph-ready support workflow")

engine = OrderEngine()
agent = SupportAgent()

a,b,c,d = st.columns(4)
a.metric("Orders", f"{len(engine.df):,}")
b.metric("Return rate", f"{engine.df.returned.mean():.2%}")
c.metric("Avg price", f"₹{engine.df.price_inr.mean():,.0f}")
d.metric("COD share", f"{(engine.df.payment_method=='COD').mean():.2%}")

left,right = st.columns([1,2])
with left:
    st.subheader("Order intelligence")
    oid = st.number_input("Order ID", 1, int(engine.df.order_id.max()), 1)
    if st.button("Analyze order", use_container_width=True):
        result = engine.order_summary(oid)
        st.json(result)
with right:
    st.subheader("Support assistant")
    if "messages" not in st.session_state: st.session_state.messages=[]
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
    prompt = st.chat_input("Ask about an order, return risk, delivery or policy")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        response = agent.chat(prompt)
        st.session_state.messages.append({"role":"assistant","content":response["answer"]})
        st.rerun()

st.subheader("Return intelligence")
st.dataframe(engine.df.groupby("product_category").returned.agg(["count","mean"]).rename(columns={"mean":"return_rate"}), use_container_width=True)
