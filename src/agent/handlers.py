# src/agent/handlers.py
import pandas as pd

FORECAST = "data/processed/forecast_alerts.csv"

def _df():
    df = pd.read_csv(FORECAST)
    df.columns = df.columns.str.strip().str.lower()
    return df

def list_hospitals():
    df = _df()
    hospitals = sorted(df["hospital_id"].unique().tolist())
    if not hospitals:
        return "No hospitals found."
    return "Hospitals:\n- " + "\n- ".join(hospitals)

def highest_urgency():
    df = _df()
    # use last forecast_date
    d0 = str(pd.to_datetime(df["forecast_date"]).max().date())
    day = df[df["forecast_date"] == d0]
    if day.empty: return "No forecast rows."
    top = day.sort_values("urgency_score", ascending=False).head(1)
    r = top.iloc[0]
    return f"Highest urgency: {r['hospital_id']} (score {r['urgency_score']})."

def lowest_urgency():
    df = _df()
    d0 = str(pd.to_datetime(df["forecast_date"]).max().date())
    day = df[df["forecast_date"] == d0]
    if day.empty: return "No forecast rows."
    r = day.sort_values("urgency_score", ascending=True).head(1).iloc[0]
    return f"Lowest urgency: {r['hospital_id']} (score {r['urgency_score']})."

def capacity_lookup(hospital_id: str | None):
    df = _df()
    d0 = str(pd.to_datetime(df["forecast_date"]).max().date())
    day = df[df["forecast_date"] == d0]
    if hospital_id:
        row = day[day["hospital_id"].str.lower() == hospital_id.lower()]
        if row.empty:
            return f"No data for {hospital_id} on {d0}."
        r = row.iloc[0]
        return (f"{hospital_id} on {d0}: ICU free {r['pred_free_icu']}, "
                f"Beds free {r['pred_free_beds']}, Urgency {r['urgency_score']}.")
    # no hospital → summary
    parts = []
    for _, r in day.sort_values("hospital_id").iterrows():
        parts.append(f"- {r['hospital_id']}: ICU {r['pred_free_icu']}, Beds {r['pred_free_beds']}, U {r['urgency_score']}")
    return "Capacities on " + d0 + ":\n" + "\n".join(parts)

def rank_metric(metric: str):
    df = _df()
    d0 = str(pd.to_datetime(df["forecast_date"]).max().date())
    day = df[df["forecast_date"] == d0]
    if metric == "icu":
        s = day.sort_values("pred_free_icu", ascending=True)  # shortage first
        rows = [f"{i+1}. {r['hospital_id']} (ICU free: {r['pred_free_icu']})"
                for i, r in s.iterrows()]
        return "ICU ranking (low→high free):\n" + "\n".join(rows[:10])
    if metric == "beds":
        s = day.sort_values("pred_free_beds", ascending=True)
        rows = [f"{i+1}. {r['hospital_id']} (Beds free: {r['pred_free_beds']})"
                for i, r in s.iterrows()]
        return "Beds ranking (low→high free):\n" + "\n".join(rows[:10])
    # default urgency
    s = day.sort_values("urgency_score", ascending=False)
    rows = [f"{i+1}. {r['hospital_id']} (Urgency score: {r['urgency_score']})"
            for i, r in s.iterrows()]
    return "Urgency ranking (high→low):\n" + "\n".join(rows[:10])
