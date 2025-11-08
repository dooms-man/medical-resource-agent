# 🏥 Medical Resource Allocation AI Agent

A **hybrid AI system** designed to forecast ICU/bed shortages, optimize patient transfers, and interact via an intelligent conversational agent.  
The system combines **LLMs, RAG, deterministic logic, forecasting, and feedback-driven learning** to support hospital decision-making.

---

## ✅ 1. Problem Approach

Hospitals often suffer from:

- ICU/bed imbalances  
- Staff shortages  
- Uneven resource distribution across cities  
- Lack of explainability behind transfer decisions  

### 🔍 Our Solution

A **hybrid AI approach** with four main components:

1. **Forecasting** of ICU/bed shortages  
2. **Optimization** of patient transfers using cost-aware LP  
3. **Conversational Agent** with:
   - Strict intent routing  
   - RAG retrieval  
   - LLM reasoning  
4. **Adaptive Learning** from user feedback (binary + comment-based)

This ensures the system remains:
> ✅ Accurate • ✅ Explainable • ✅ Non-hallucinatory • ✅ User-adaptive

---

## ✅ 2. Data Sources Used

| File | Purpose |
|------|----------|
| `hospital_timeseries.csv` | 30-day bed/ICU/staff trends for each hospital |
| `geo_costs.csv` | Pairwise hospital distances for optimizer |
| `forecast_alerts.csv` | Forecast outputs with urgency scores |
| `allocation_plan.csv` | Optimized transfer plan |
| `ml_default_weights.json` | Adaptive ML weights (ICU/bed/staff importance) |
| `faiss_index/` | Vector index for RAG retrieval |
| `feedback.json` | User feedback history |

📁 *All data is stored locally — no external APIs used.*

---

## ✅ 3. Agent Architecture & Design Choices

### 🧩 A. Intent Router (Zero Hallucination Layer)

A **rule-based classifier** routes structured queries to deterministic handlers:

| Example Query | Routed To |
|----------------|-----------|
| “list hospitals” | list handler |
| “highest urgency” | alert handler |
| “ICU capacity of Pune” | ICU handler |

> Prevents hallucinations and guarantees correctness.

---

### 📚 B. RAG Retrieval Layer

Used for explanatory or analytical questions:

- Retrieves relevant context from **FAISS**
- Uses **MiniLM embeddings** for high recall
- LLM answers **only** using retrieved context  

➡️ Avoids hallucination in open-ended queries.

---

### 🧠 C. LLM Reasoning Layer

- Model: **Groq LLaMA-3 8B Instant**  
- Used for: explanation, justification, natural summaries  
- Generates responses **constrained by retrieved context**

---

### 🔁 D. Adaptive Learning Layer

Learns user preferences from:

- ✅ Keywords in comments  
- ✅ Helpful / Not Helpful feedback  
- ✅ Session-specific patterns  

Adapts:
- ICU/bed/staff weight importance  
- Response verbosity  
- Reasoning detail  

> 🧩 Fulfills requirement:  
> *“Agent must evolve understanding of user priorities without manual tuning.”*

---

## ✅ 4. Binary Feedback System (Helpful / Not Helpful)

Users can react to every AI message:

| Signal | Behavior |
|---------|-----------|
| ✅ Helpful | Agent replies shorter & more direct |
| ❌ Not Helpful | Agent replies more detailed, step-by-step |

Feedback is stored in `feedback.json` with timestamps.

### 💡 Why This Matters

Enables:
- User preference learning  
- Adaptive behavior  
- Reasoning evolution  
- Quantitative evaluation  

And improves:
> ✅ User modeling • ✅ Adaptive AI • ✅ Explainability • ✅ Continuous improvement

---

## ✅ 5. Logic Behind Adaptive Learning & Reasoning

### ⚙️ A. Weight Adaptation

User comments like:
> “Focus on beds” or “ICU is more important”

→ Automatically shift weights in `ml_default_weights.json`  
→ Alters urgency scores → affects optimizer → changes recommendations.

---

### 💬 B. Response Style Adaptation

- More downvotes → more detailed explanations  
- More upvotes → concise, focused replies  

Demonstrates:
> 🧠 Behavioral adaptation and user-centered design

---

### 🔗 C. Hybrid Reasoning Design

Combines:
- **Deterministic logic** (accuracy)
- **LLM reasoning** (flexibility)

Ensures:
> ✅ Zero hallucination on structured tasks  
> ✅ Natural, high-quality explanations

---

## ⚠️ 6. Limitations

- No real-time hospital APIs (CSV-based only)  
- Intent detection is rule-based (not ML)  
- FAISS may return irrelevant chunks for edge cases  
- Session memory not yet persisted (no Redis)  
- No streaming responses  
- Weight learning is heuristic, not ML-based

---

## 🚀 7. Future Extensions

Planned improvements:
(I'm learning Langgraph)
- ✅ Redis-based 10-min session memory  
- ✅ ML-based intent classification  
- ✅ LangGraph multi-agent roles  
- ✅ Reinforcement learning for weight updates  
- ✅ Real-time dashboards  
- ✅ Integration with hospital APIs  
- ✅ Streaming responses with Groq

---


