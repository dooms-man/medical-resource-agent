# src/agent/urgency.py
import json, os

WEIGHTS_JSON = "data/processed/ml_default_weights.json"

def load_ml_weights():
    if os.path.exists(WEIGHTS_JSON):
        with open(WEIGHTS_JSON, "r") as f:
            w = json.load(f)
        return dict(
            icu_weight=float(w.get("icu_weight", 0.6)),
            bed_weight=float(w.get("bed_weight", 0.4)),
            staff_weight=float(w.get("staff_weight", 0.0)),
        )
    return dict(icu_weight=0.6, bed_weight=0.4, staff_weight=0.0)
