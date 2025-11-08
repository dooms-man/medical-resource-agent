import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

INP = "data/raw/hospital_timeseries.csv"
OUT = "data/processed/ml_default_weights.json"

def normalize(x):
    x = np.clip(x, 0, None)
    s = x.max()
    return x / s if s > 0 else x

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_csv(INP)

    # Shortage percents
    bed_short = (df["occupied_beds"] - (0.9 * df["total_beds"])).clip(lower=0) / df["total_beds"].replace(0, np.nan)
    bed_short = bed_short.fillna(0)

    icu_short = (df["occupied_icu"] - (0.85 * df["total_icu"])).clip(lower=0) / df["total_icu"].replace(0, np.nan)
    icu_short = icu_short.fillna(0)

    staff_gap = (df["staff_required"] - df["staff_available"]).clip(lower=0)
    staff_gap_pct = staff_gap / (df["staff_required"].replace(0, np.nan))
    staff_gap_pct = staff_gap_pct.fillna(0)

    # Synthetic outcome to learn from: higher when pressure is high
    # We include admissions and discharges dynamics to add realism
    adm = df["admissions"].astype(float)
    dis = df["discharges"].astype(float)
    turnover = (adm - dis).clip(lower=0)
    turnover_n = normalize(turnover.values)

    X = np.vstack([
        icu_short.values,
        bed_short.values,
        staff_gap_pct.values
    ]).T
    # Ground-truth we pretend exists (unknown in reality) + noise
    y = (
        0.55 * icu_short.values
      + 0.30 * bed_short.values
      + 0.15 * staff_gap_pct.values
      + 0.05 * turnover_n
      + np.random.normal(0, 0.02, size=len(df))
    )

    model = LinearRegression()
    model.fit(X, y)
    coefs = np.abs(model.coef_)  # importance >= 0
    if coefs.sum() == 0:
        coefs = np.array([0.5, 0.3, 0.2])
    weights = (coefs / coefs.sum()).tolist()

    out = {
        "icu_weight":   round(weights[0], 4),
        "bed_weight":   round(weights[1], 4),
        "staff_weight": round(weights[2], 4),
        "note": "learned from synthetic strain proxy via linear regression"
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"✅ Saved ML default weights → {OUT}: {out}")

if __name__ == "__main__":
    main()
