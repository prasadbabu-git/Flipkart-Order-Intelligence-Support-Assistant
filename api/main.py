from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from core.agent import SupportAgent

app = FastAPI(title="Flipkart Order Intelligence API", version="1.0.0")
agent = SupportAgent()

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    order_id: int | None = None

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest): return agent.chat(req.message, req.order_id)

@app.get("/orders/{order_id}")
def order(order_id: int):
    try: return agent.orders.order_summary(order_id)
    except KeyError as e: raise HTTPException(404, str(e))
