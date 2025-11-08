from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.feedback_store import store_feedback
from src.rag.query_agent_chat import answer_with_explanation
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatReq(BaseModel):
    session_id: str
    message: str

class FeedbackReq(BaseModel):
    session_id: str
    signal: str        # "up" or "down"
    comment: str = ""  # optional

@app.post("/chat")
def chat(req: ChatReq):
    response = answer_with_explanation(req.session_id, req.message)
    return {"response": response}

@app.post("/feedback")
def feedback(req: FeedbackReq):
    try:
        store_feedback(req.session_id, req.signal, req.comment)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

