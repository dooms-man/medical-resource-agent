# src/agent/faq.py
FAQ = {
    "what is urgency": "Urgency is a normalized shortage signal combining ICU and beds. Higher = more shortage pressure.",
    "how are transfers chosen": "A linear program minimizes distance cost while satisfying shortages, bounded by supply."
}

def faq_lookup(q: str) -> str | None:
    s = q.lower().strip()
    for k, v in FAQ.items():
        if k in s: return v
    return None
