"""
query_agent.py
--------------
Core single-turn RAG engine for the Medical Resource Allocation Agent.
Provides a reusable `answer_query()` function.

Automatically cleans CSV columns to avoid KeyErrors.
Compatible with LangChain 1.0.3.
"""

import os
import argparse
import pandas as pd

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# Helper: Read CSV with CLEANED column names
# ============================================================

def read_clean_csv(path):
    df = pd.read_csv(path)
    # Fix ALL column naming problems
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("__", "_")
    )
    return df


# ============================================================
# Build Text Corpus (Forecast + Allocation)
# ============================================================

def build_text_corpus():
    corpus = []

    # ---------------- Forecast Alerts ----------------
    fore_path = "data/processed/forecast_alerts.csv"
    if os.path.exists(fore_path):
        df = read_clean_csv(fore_path)

        for _, r in df.iterrows():
            hospital = r.get("hospital_id") or r.get("name") or "Unknown Hospital"
            date = r.get("forecast_date", "Unknown Date")

            corpus.append(
                f"Forecast for {hospital} on {date}: "
                f"free ICU={r.get('pred_free_icu')}, "
                f"free beds={r.get('pred_free_beds')}, "
                f"urgency={r.get('urgency_score')}."
            )

    # ---------------- Allocation Plan ----------------
    alloc_path = "data/processed/allocation_plan.csv"
    if os.path.exists(alloc_path):
        df = read_clean_csv(alloc_path)

        for _, r in df.iterrows():
            src = r.get("from") or r.get("source") or r.get("sender") or "Unknown"
            dst = r.get("to") or r.get("destination") or r.get("receiver") or "Unknown"

            resource = r.get("resource", "resource")
            qty = r.get("quantity", 1)
            dist = r.get("distance_km", 0)

            corpus.append(
                f"Transfer: {src} -> {dst} | {qty} {resource} | {dist} km."
            )

    return corpus


# ============================================================
# Load or Build FAISS Vectorstore
# ============================================================

def get_vectorstore():
    corpus = build_text_corpus()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    path = "data/processed/faiss_index"

    # Load existing
    if os.path.exists(os.path.join(path, "index.faiss")):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    # Build new
    vs = FAISS.from_texts(corpus, embeddings)
    vs.save_local(path)
    return vs


# ============================================================
# Core RAG Engine: answer_query()
# ============================================================

def answer_query(question: str) -> str:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Retrieve documents
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])

    # LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Prompt Template
    prompt = PromptTemplate.from_template("""
You are a medical resource allocation AI assistant.
Use ONLY the given context to answer.

Context:
{context}

Question:
{question}
""")

    final = prompt.format(context=context, question=question)
    response = llm.invoke(final)
    return response.content


# ============================================================
# CLI Runner
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="Summarize all transfers.")
    args = parser.parse_args()

    print("\n=== 💬 Answer ===")
    print(answer_query(args.query))
