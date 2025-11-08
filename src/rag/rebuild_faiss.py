import os
import shutil
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

FORECAST = "data/processed/forecast_alerts.csv"
ALLOC = "data/processed/allocation_plan.csv"
FAISS_DIR = "data/processed/faiss_index"

def main():
    texts = []

    if os.path.exists(FORECAST):
        df = pd.read_csv(FORECAST)
        for _, r in df.iterrows():
            hosp = r.get("hospital_id")
            date = r.get("forecast_date")
            icu = r.get("pred_free_icu")
            beds = r.get("pred_free_beds")
            urg = r.get("urgency_score")
            texts.append(f"Forecast: On {date}, {hosp} expects {icu} free ICU and {beds} free beds. Urgency {urg}.")

    if os.path.exists(ALLOC):
        df = pd.read_csv(ALLOC)
        for _, r in df.iterrows():
            texts.append(
                f"Transfer plan: {r.get('from')} -> {r.get('to')} | "
                f"{r.get('quantity')} {r.get('resource')} | {r.get('distance_km')} km on {r.get('date')}."
            )

    if not texts:
        print("No texts to index.")
        return

    if os.path.exists(FAISS_DIR):
        shutil.rmtree(FAISS_DIR)

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_texts(texts, emb)
    vs.save_local(FAISS_DIR)
    print(f"✅ Rebuilt FAISS → {FAISS_DIR} with {len(texts)} chunks.")

if __name__ == "__main__":
    main()
