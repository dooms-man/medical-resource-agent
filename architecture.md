```mermaid
flowchart LR
    UI["Frontend (HTML/JS Chat UI)"]
    API["FastAPI Server"]
    AGENT["Routing Layer (Intent Classifier + FAQ + Rule Handlers)"]
    RAG["RAG Engine (Prompt + Groq LLM)"]
    RET["FAISS Vector Store"]
    FEED["Feedback Store (JSON-based)"]
    PIPE["Data Pipeline (Forecast + Optimization + FAISS Rebuild)"]
    DATA["Processed Data (CSV Files)"]

    UI -->|/chat| API
    API --> AGENT
    AGENT -->|structured intent| RAG
    AGENT -->|FAQ hit| UI
    AGENT -->|handler response| UI
    RAG --> RET
    RAG --> UI

    UI -->|feedback| API --> FEED

    PIPE --> DATA
    PIPE --> RET


This describes your exact system: UI → FastAPI → Router → RAG → FAISS, plus pipeline + feedback.

---

# ✅ **2. Detailed Pipeline Diagram (Mermaid)**  
Use this for `pipeline.md`:

```md
```mermaid
flowchart TD
    RAW["Raw Timeseries CSV\n(hospital_timeseries.csv)"]
    GEO["Geo Cost CSV\n(geo_costs.csv)"]

    CLEAN["Normalize & Clean Columns"]
    FORECAST["Forecasting Module\n(Prophet → fallback baseline)"]
    OPT["Linear Optimization (PuLP)\nICU + Beds transfer plan"]
    MERGE["Generate Forecast Alerts CSV"]

    FAISS["Rebuild FAISS Embeddings\n(sentence-transformer)"]
    OUT1["forecast_alerts.csv"]
    OUT2["allocation_plan.csv"]
    OUT3["faiss_index/"]

    RAW --> CLEAN --> FORECAST --> MERGE --> OUT1
    GEO --> OPT
    MERGE --> OPT --> OUT2
    OUT1 --> FAISS --> OUT3
    OUT2 --> FAISS


