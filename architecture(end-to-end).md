                   ┌─────────────────────────────┐
                   │    Hospital Timeseries CSV   │
                   └────────────┬────────────────┘
                                │
                                ▼
                     (A) Forecasting Pipeline
      Prophet / Baseline Model → predict free ICU & beds for next n days
                                │
                                ▼
                    forecast_alerts.csv (structured data)

                                │
                                ▼
                     (B) Optimizer (PuLP, LP model)
         shortage & surplus analysis → distance-aware resource transfers

                                │
                                ▼
                    allocation_plan.csv (structured data)

                                │
                                ▼
                     (C) Vector Index Builder (FAISS)
         Convert forecasts + transfers → text chunks → embeddings

                                │
                                ▼
                      FAISS Vectorstore (K=4 retrieval)

    ┌───────────────────────────────────────────────────────────────────────┐
    │                                                                       │
    │  (D) Agent Layer (Hybrid Reasoning Engine)                             │
    │                                                                       │
    │   ┌────────────────────────┐        ┌───────────────────────────┐     │
    │   │  Intent Classifier     │        │   FAQ Fast-Path Engine    │     │
    │   └──────────┬─────────────┘        └─────────────┬────────────┘     │
    │              │                                    │                   │
    │              ▼                                    ▼                   │
    │   Structured Handler Routes         If known question → direct answer  │
    │   (list hospitals, rank ICU, etc.)                                    │
    │                                                                       │
    │                    ┌──────────────────────────────────────┐           │
    │                    │              RAG Engine               │           │
    │                    │ Retrieve chunks → LLM answers         │           │
    │                    └──────────────────────────────────────┘           │
    │                                                                       │
    │         (E) Feedback Engine                                           │
    │    thumbs-up/down + comment → keyword-based weight update             │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘

                                │
                                ▼
                          Final Output
     ✔ ICU / bed shortage predictions  
     ✔ Optimized transfer plan  
     ✔ Hospital ranking  
     ✔ Explainability trace  
     ✔ Adapted + personalized outputs
