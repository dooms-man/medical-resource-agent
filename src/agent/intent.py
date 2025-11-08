# src/agent/intent.py
import re

def detect_intent(q: str) -> str:
    s = q.lower().strip()
    if re.search(r"\blist\b.*\bhospitals?\b", s): return "list_hospitals"
    if "highest urgency" in s or "top urgency" in s: return "highest_urgency"
    if "lowest urgency" in s: return "lowest_urgency"
    if "capacity" in s or "free icu" in s or "free beds" in s: return "capacity_lookup"
    if "rank" in s or "ranking" in s: return "rank_metric"
    if "explain" in s or "why" in s or "reason" in s: return "explain"
    return "none"
