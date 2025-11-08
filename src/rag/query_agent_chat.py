# src/rag/query_agent_chat.py
import sys, os, re, argparse, uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from agent.intent import detect_intent
from agent.handlers import list_hospitals, highest_urgency, lowest_urgency, capacity_lookup, rank_metric
from agent.faq import faq_lookup
from agent.feedback_store import load_recent_feedback
from agent.explain import explain

def _retriever():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local("data/processed/faiss_index", emb, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": 4})

LLM = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
PROMPT = PromptTemplate(
    input_variables=["context","question"],
    template="You are a medical resource assistant. Use ONLY the context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer briefly and factually."
)

def _rag(q: str) -> str:
    retriever = _retriever()
    docs = retriever.invoke(q)
    ctx = "\n".join(str(d.page_content) for d in docs)
    return LLM.invoke(PROMPT.format(context=ctx, question=q)).content

SESSIONS = {}
def get_session_history(session_id: str):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = InMemoryChatMessageHistory()
    return SESSIONS[session_id]

MEDICAL_KEYWORDS = [
    "icu","bed","beds","ventilator","ventilators",
    "hospital","forecast","shortage","urgency","transfer",
    "resource","allocation","capacity","explain","reason","why"
]
DEFAULT_DENIAL = ("I only answer medical resource allocation queries such as ICU/bed forecasts, "
                  "hospital urgency, and transfer recommendations.")

DEFAULT_PROMPT = """
You are a Medical Resource Allocation AI Agent.
You must use ONLY the retrieved context to answer.
Respond concisely and professionally.

Context:
{context}

Question: {question}

Answer:
"""

def build_adaptive_prompt(session_id: str):
    fb = load_recent_feedback(session_id)
    if not fb: return DEFAULT_PROMPT
    pos = sum(1 for x in fb if x.get("signal") == "up")
    neg = sum(1 for x in fb if x.get("signal") == "down")
    hint = "User prefers detailed reasoning." if neg > pos else "User prefers short, clear summaries."
    return DEFAULT_PROMPT + f"\n\n### User Preference Hint:\n{hint}\n"

def load_vectorstore():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local("data/processed/faiss_index", emb, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": 4})

def build_chain(session_id: str):
    retriever = load_vectorstore()
    prompt_text = build_adaptive_prompt(session_id)
    prompt = PromptTemplate(input_variables=["context","question"], template=prompt_text)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

    def rag_fn(inputs):
        q = inputs["question"]
        if isinstance(q, list):
            q = " ".join(str(m.content if hasattr(m, "content") else m) for m in q)
        q_low = q.lower()

        if not any(k in q_low for k in MEDICAL_KEYWORDS):
            return {"answer": DEFAULT_DENIAL}

        docs = retriever.invoke(q)
        ctx = "\n".join([str(d.page_content) for d in docs])
        final_prompt = prompt.format(context=ctx, question=q)
        ans = llm.invoke(final_prompt).content
        return {"answer": ans}

    wrapped = RunnableLambda(rag_fn)
    return RunnableWithMessageHistory(
        runnable=wrapped,
        get_session_history=get_session_history,
        input_messages_key="question",
        output_messages_key="answer",
    )

def answer_with_explanation(session_id: str, question: str) -> str:
    # FAQ short-circuit
    faq = faq_lookup(question)
    if faq: return faq

    # Intent routing
    intent = detect_intent(question)
    if intent == "list_hospitals":  return list_hospitals()
    if intent == "highest_urgency": return highest_urgency()
    if intent == "lowest_urgency":  return lowest_urgency()
    if intent == "capacity_lookup":
        m = re.search(r"(?:for|at|of)\s+([a-z0-9\-]+)$", question.lower())
        return capacity_lookup(m.group(1) if m else None)
    if intent == "rank_metric":
        q = question.lower()
        metric = "urgency"
        if "icu" in q: metric = "icu"
        elif "bed" in q: metric = "beds"
        return rank_metric(metric)
    if intent == "explain":
        narrative = _rag(question)
        trace = explain(date=None, trace_k=3)
        return f"{narrative}\n\n---\n📊 Explainability:\n{trace}"

    # Default RAG
    chain = build_chain(session_id)
    res = chain.invoke({"question": question}, config={"session_id": session_id})
    return res["answer"]

def main(query: str):
    sid = "chat-" + str(uuid.uuid4())
    print(answer_with_explanation(sid, query))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    args = ap.parse_args()
    main(args.query)
