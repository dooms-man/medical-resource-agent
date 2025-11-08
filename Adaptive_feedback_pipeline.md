                  ┌────────────────────────────────┐
                  │     Hospital Timeseries CSV     │
                  └───────────────┬────────────────┘
                                  │
                                  ▼
                     (1) Preprocessing + Normalization
                                  │
                                  ▼
                     (2) Linear Regression Training
          ┌─────────────────────────────┬──────────────────────────────┐
          │                              │                              │
       ICU Usage → coefficient → icu_weight (base)                     │
       Bed Usage → coefficient → bed_weight (base)                     │
       Staff Gap → coefficient → staff_weight (base)                  │
          └─────────────────────────────┴──────────────────────────────┘
                                  │
                                  ▼
                       Save weights as ML defaults
                                  │
                                  ▼
                     ┌──────────────────────────────┐
                     │  User Interacts With Agent   │
                     └───────────────┬──────────────┘
                                     │
                     User Feedback (thumbs / comment)
                                     │
                                     ▼
                         (3) Feedback Keyword Parser
                   ┌───────────────┬───────────────┬───────────────┐
                   │               │               │               │
               "icu" → bump icu_weight        "bed"→ bump bed_weight  
              "staff"→ bump staff_weight       ... etc.
                   └───────────────┴───────────────┴───────────────┘
                                     │
                                     ▼
                         (4) Normalize All Weights
                                     │
                                     ▼
                         Updated Weight Vector (Soft ML)
                                     │
                                     ▼
                     ┌──────────────────────────────────┐
                     │ Recompute Urgency for Each Hosp  │
                     └────────────────┬──────────────────┘
                                      │
                                      ▼
                 urgency = w_ICU*icu_short + w_Bed*bed_short + w_Staff*staff_short
                                      │
                                      ▼
                           Push Updated Urgencies to FAISS
                                      │
                                      ▼
                     RAG Querying → Better Priority-Aware Answers
