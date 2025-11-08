import json
from pathlib import Path
from datetime import datetime

FEEDBACK_PATH = Path("data/processed/feedback.json")

def _load_all():
    if FEEDBACK_PATH.exists():
        with open(FEEDBACK_PATH, "r") as f:
            return json.load(f)
    return {}

def _save_all(data):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(data, f, indent=2)

def store_feedback(session_id: str, signal: str, comment: str = ""):
    """Store feedback for a session. signal = 'up' or 'down'."""
    data = _load_all()

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "signal": signal,
        "comment": comment
    }

    if session_id not in data:
        data[session_id] = []

    data[session_id].append(entry)
    _save_all(data)

def load_recent_feedback(session_id: str, limit: int = 20):
    """Return recent feedback entries for a session."""
    data = _load_all()
    return data.get(session_id, [])[-limit:]
