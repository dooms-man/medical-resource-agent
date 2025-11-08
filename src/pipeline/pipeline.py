"""
pipeline.py — Patched with slack variables so LP never dies.

End-to-end pipeline:
1) Forecast shortages (Prophet if available → fallback to baseline)
2) Optimize transfers with distance-aware LP + UNMET DEMAND SLACK
3) Save processed outputs
4) Rebuild FAISS index for the RAG agent

Run:
    python src/pipeline/pipeline.py

Outputs:
    data/processed/forecast_alerts.csv
    data/processed/allocation_plan.csv
    data/processed/faiss_index/   (freshly rebuilt)
"""

import os
import sys
import shutil
import math
import warnings
from datetime import datetime, timedelta

import pandas as pd

# Optional deps
try:
    from prophet import Prophet  # pip install prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

# Optimizer
try:
    import pulp
except Exception as e:
    raise RuntimeError("PuLP not installed. Install with: pip install pulp") from e

# Embeddings / Vector DB for FAISS rebuild
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception as e:
    raise RuntimeError("langchain-community not installed. Install with: pip install langchain-community sentence-transformers faiss-cpu") from e


# ------------------------ Config ------------------------

RAW_TIMESERIES = "data/processed/hospital_timeseries.csv"  # your master timeseries
GEO_COSTS = "data/processed/geo_costs.csv"                 # pairwise distances
OUT_DIR = "data/processed"
FORECAST_OUT = os.path.join(OUT_DIR, "forecast_alerts.csv")
ALLOC_OUT = os.path.join(OUT_DIR, "allocation_plan.csv")
FAISS_DIR = os.path.join(OUT_DIR, "faiss_index")

FORECAST_HORIZON_DAYS = 3
URGENCY_W_BEDS = 0.4
URGENCY_W_ICU = 0.6

# objective weights
DIST_COST_WEIGHT = 0.02      # small penalty per km
SLACK_PENALTY = 1000.0       # huge penalty for unmet demand so solver prefers transfers
L2_REG = 1e-4                # tiny regularizer on flows

RANDOM_SEED = 42


# ------------------------ Helpers ------------------------

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("__", "_")
    )
    return df

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _read_csv_clean(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    return _clean_cols(df)

def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    return default


# ------------------------ 1) Forecasting ------------------------

def _baseline_forecast_last_k(df_grp: pd.DataFrame, date_col: str, target: str, horizon: int) -> pd.DataFrame:
    """Simple baseline: carry last value forward as prediction."""
    df_grp = df_grp.sort_values(date_col)
    last_val = float(df_grp[target].iloc[-1])
    last_date = pd.to_datetime(df_grp[date_col].iloc[-1])
    rows = []
    for i in range(1, horizon + 1):
        d = (last_date + timedelta(days=i)).date().isoformat()
        rows.append({"forecast_date": d, "pred": float(last_val)})
    return pd.DataFrame(rows)

def _prophet_forecast(df_grp: pd.DataFrame, date_col: str, target: str, horizon: int) -> pd.DataFrame:
    """Prophet forecast for a single (hospital, metric) series."""
    mdl = Prophet(seasonality_mode="multiplicative", yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    tmp = df_grp[[date_col, target]].rename(columns={date_col: "ds", target: "y"}).copy()
    tmp["ds"] = pd.to_datetime(tmp["ds"])
    mdl.fit(tmp)
    future = mdl.make_future_dataframe(periods=horizon, freq="D", include_history=False)
    fc = mdl.predict(future)[["ds", "yhat"]]
    fc = fc.rename(columns={"ds": "forecast_date", "yhat": "pred"})
    fc["forecast_date"] = fc["forecast_date"].dt.date.astype(str)
    return fc

def forecast_all():
    df = _read_csv_clean(RAW_TIMESERIES)

    # Expected flexible columns:
    # date, hospital_id/name, total_beds, occupied_beds, total_icu, occupied_icu
    df["date"] = pd.to_datetime(
        _coalesce(df.get("date"), df.get("day"), df.get("ds"), default=pd.Series(dtype="datetime64[ns]"))
    )
    if df["date"].isna().all():
        raise ValueError("Could not parse a date column in hospital_timeseries.csv")

    df["hospital_id"] = _coalesce(df.get("hospital_id"), df.get("name"), default=None)
    if df["hospital_id"] is None or df["hospital_id"].isna().all():
        raise ValueError("hospital_timeseries.csv must contain 'hospital_id' or 'name'.")

    # Robust free capacities
    total_beds = _coalesce(df.get("total_beds"), df.get("beds_capacity"), default=0)
    occ_beds   = _coalesce(df.get("occupied_beds"), default=0)
    total_icu  = _coalesce(df.get("total_icu"), df.get("icu_capacity"), default=0)
    occ_icu    = _coalesce(df.get("occupied_icu"), default=0)

    df["free_beds"] = (total_beds - occ_beds).astype(float)
    df["free_icu"]  = (total_icu - occ_icu).astype(float)

    # Forecast per hospital
    out_rows = []
    for hosp, g in df.groupby("hospital_id"):
        g = g.sort_values("date")

        # Pick model per series
        try:
            if _HAS_PROPHET and len(g) >= 7:
                fore_beds = _prophet_forecast(g, "date", "free_beds", FORECAST_HORIZON_DAYS)
                fore_icu  = _prophet_forecast(g, "date", "free_icu",  FORECAST_HORIZON_DAYS)
            else:
                fore_beds = _baseline_forecast_last_k(g, "date", "free_beds", FORECAST_HORIZON_DAYS)
                fore_icu  = _baseline_forecast_last_k(g, "date", "free_icu",  FORECAST_HORIZON_DAYS)
        except Exception as e:
            warnings.warn(f"Prophet failed for {hosp} ({e}); using baseline.")
            fore_beds = _baseline_forecast_last_k(g, "date", "free_beds", FORECAST_HORIZON_DAYS)
            fore_icu  = _baseline_forecast_last_k(g, "date", "free_icu",  FORECAST_HORIZON_DAYS)

        merged = pd.merge(fore_beds, fore_icu, on="forecast_date", suffixes=("_beds", "_icu"))
        merged.insert(0, "hospital_id", hosp)
        merged = merged.rename(columns={"pred_beds": "pred_free_beds", "pred_icu": "pred_free_icu"})

        # clamp extremes a bit to avoid absurd negatives that nuke LP
        merged["pred_free_beds"] = merged["pred"].iloc[:, 0] if "pred" in merged.columns else merged["pred_free_beds"]
        if "pred_free_beds" not in merged.columns:
            merged = merged.rename(columns={"pred_beds": "pred_free_beds"})

        if "pred_free_icu" not in merged.columns:
            merged = merged.rename(columns={"pred_icu": "pred_free_icu"})

        merged["pred_free_beds"] = merged["pred_free_beds"].astype(float).clip(lower=-20, upper=1e6)
        merged["pred_free_icu"]  = merged["pred_free_icu"].astype(float).clip(lower=-10, upper=1e6)

        # shortage flags and urgency
        pf_beds = merged["pred_free_beds"]
        pf_icu  = merged["pred_free_icu"]

        merged["shortage_beds"] = (pf_beds < 0).astype(int)
        merged["shortage_icu"]  = (pf_icu  < 0).astype(int)

        # urgency ~ weighted magnitude of shortages only
        urg = (URGENCY_W_ICU * (-pf_icu.clip(upper=0)) + URGENCY_W_BEDS * (-pf_beds.clip(upper=0)))
        if urg.max() > 0:
            urg = urg / urg.max()
        merged["urgency_score"] = urg.round(3)

        # keep only expected columns
        merged = merged[["hospital_id", "forecast_date", "pred_free_beds", "pred_free_icu", "shortage_beds", "shortage_icu", "urgency_score"]]
        out_rows.append(merged)

    out = pd.concat(out_rows, ignore_index=True)
    _ensure_dir(OUT_DIR)
    out.to_csv(FORECAST_OUT, index=False)
    return out


# ------------------------ 2) Optimization (with slack) ------------------------

def load_distances():
    dist = _read_csv_clean(GEO_COSTS)
    # Accept common schemas: from/to or source/destination
    src_col = "from" if "from" in dist.columns else ("source" if "source" in dist.columns else None)
    dst_col = "to" if "to" in dist.columns else ("destination" if "destination" in dist.columns else None)
    if not src_col or not dst_col or "distance_km" not in dist.columns:
        raise ValueError("geo_costs.csv must have 'from'/'to' (or 'source'/'destination') and 'distance_km' columns.")
    return dist.rename(columns={src_col: "from", dst_col: "to"})

def build_supply_demand(fore_df: pd.DataFrame, target_date: str):
    """Compute total surplus/shortage per hospital for a specific forecast_date."""
    fore_df = fore_df.copy()
    fore_df["forecast_date"] = pd.to_datetime(fore_df["forecast_date"]).dt.date.astype(str)

    if target_date == "latest":
        latest = pd.to_datetime(fore_df["forecast_date"]).max().date()
        d0 = latest.isoformat()
    else:
        # normalize incoming date
        d0 = pd.to_datetime(target_date).date().isoformat()

    # normalize forecast dates
    forecast_dates = set(pd.to_datetime(fore_df["forecast_date"]).dt.date.astype(str))

    if d0 not in forecast_dates:
        raise ValueError(
            f"Requested date {d0} not found. Available: {sorted(list(forecast_dates))}"
        )


    day = fore_df[fore_df["forecast_date"] == d0].copy()

    supplies = {}
    demands = {}
    for _, r in day.iterrows():
        h = r["hospital_id"]
        supplies.setdefault(h, {"beds": 0.0, "icu": 0.0})
        demands.setdefault(h, {"beds": 0.0, "icu": 0.0})
        # positive is supply, negative becomes demand
        beds = float(r["pred_free_beds"])
        icu  = float(r["pred_free_icu"])
        supplies[h]["beds"] = max(beds, 0.0)
        supplies[h]["icu"]  = max(icu, 0.0)
        demands[h]["beds"]  = max(-beds, 0.0)
        demands[h]["icu"]   = max(-icu, 0.0)

    return supplies, demands, d0

def optimize_transfers(fore_df: pd.DataFrame, dist_df: pd.DataFrame, target_date: str):
    supplies, demands, d0 = build_supply_demand(fore_df, target_date)

    hospitals = sorted(set(list(supplies.keys()) + list(demands.keys())))
    resources = ["icu", "beds"]

    # quick sanity: if zero supply and some demand -> no transfers possible
    for res in resources:
        total_supply = sum(supplies[h][res] for h in hospitals)
        total_demand = sum(demands[h][res] for h in hospitals)
        print(f"[{d0}] Resource={res}: total_supply={total_supply:.2f} total_demand={total_demand:.2f}")
    # don’t early-return; slack will absorb unmet demand

    dist_map = {(str(r["from"]), str(r["to"])): float(r["distance_km"]) for _, r in dist_df.iterrows()}
    DEFAULT_DIST = 9999.0  # discourage missing edges but still solvable

    prob = pulp.LpProblem("Medical_Resource_Allocation", pulp.LpMinimize)

    # flow variables
    X = {}
    for i in hospitals:
        for j in hospitals:
            if i == j:
                continue
            for res in resources:
                X[(i, j, res)] = pulp.LpVariable(f"x_{res}_{i}_to_{j}", lowBound=0)

    # unmet demand slack variables (make model always feasible)
    U = {}
    for j in hospitals:
        for res in resources:
            U[(j, res)] = pulp.LpVariable(f"unmet_{res}_{j}", lowBound=0)

    # objective: distance cost + tiny regularizer + heavy penalty on unmet demand
    prob += (
        pulp.lpSum(
            (DIST_COST_WEIGHT * dist_map.get((i, j), DEFAULT_DIST)) * X[(i, j, res)]
            for i in hospitals for j in hospitals if i != j for res in resources
        )
        + L2_REG * pulp.lpSum(X[(i, j, res)] for i in hospitals for j in hospitals if i != j for res in resources)
        + SLACK_PENALTY * pulp.lpSum(U[(j, res)] for j in hospitals for res in resources)
    )

    # supply constraints: outflow <= supply
    for i in hospitals:
        for res in resources:
            prob += pulp.lpSum(X[(i, j, res)] for j in hospitals if i != j) <= supplies.get(i, {}).get(res, 0.0)

    # demand constraints: inflow + slack == demand  (== not >= to define slack precisely)
    for j in hospitals:
        for res in resources:
            prob += pulp.lpSum(X[(i, j, res)] for i in hospitals if i != j) + U[(j, res)] == demands.get(j, {}).get(res, 0.0)

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    lp_status = pulp.LpStatus[status]
    print(f"LP status: {lp_status}")
    if lp_status not in ("Optimal", "Feasible"):
        # with slack, this should basically never happen
        print("⚠️ LP did not find a feasible solution even with slack. Returning empty plan.")
        plan = pd.DataFrame(columns=["date", "from", "to", "resource", "quantity", "distance_km"])
        plan.to_csv(ALLOC_OUT, index=False)
        return plan

    rows = []
    for (i, j, res), var in X.items():
        qty = var.value() or 0.0
        if qty <= 1e-6:
            continue
        rows.append({
            "date": d0,
            "from": i,
            "to": j,
            "resource": res,
            "quantity": round(qty, 2),
            "distance_km": dist_map.get((i, j), None)
        })

    plan = pd.DataFrame(rows)
    if plan.empty:
        plan = pd.DataFrame(columns=["date", "from", "to", "resource", "quantity", "distance_km"])
    plan.to_csv(ALLOC_OUT, index=False)
    return plan


# ------------------------ 3) Rebuild FAISS ------------------------

def rebuild_faiss_index():
    """
    Build a fresh FAISS index from forecast_alerts.csv + allocation_plan.csv
    so the chat agent always answers with up-to-date context.
    """
    texts = []

    if os.path.exists(FORECAST_OUT):
        df = _read_csv_clean(FORECAST_OUT)
        for _, r in df.iterrows():
            hosp = _coalesce(r.get("hospital_id"), r.get("name"), default="Unknown Hospital")
            texts.append(
                f"Forecast: On {r.get('forecast_date')}, {hosp} expects "
                f"{r.get('pred_free_icu')} free ICU and {r.get('pred_free_beds')} free beds. "
                f"Urgency score: {r.get('urgency_score')}."
            )

    if os.path.exists(ALLOC_OUT):
        df = _read_csv_clean(ALLOC_OUT)
        for _, r in df.iterrows():
            texts.append(
                f"Transfer plan: {r.get('from')} -> {r.get('to')} | "
                f"{r.get('quantity')} {r.get('resource')} | {r.get('distance_km')} km on {r.get('date')}."
            )

    if not texts:
        print("⚠️ No texts to index; skipping FAISS rebuild.")
        return

    # rebuild
    if os.path.exists(FAISS_DIR):
        shutil.rmtree(FAISS_DIR)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_texts(texts, embeddings)
    _ensure_dir(FAISS_DIR)
    vs.save_local(FAISS_DIR)
    print(f"✅ Rebuilt FAISS index at {FAISS_DIR} with {len(texts)} chunks.")


# ------------------------ Main ------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default="latest",
                    help="Forecast date to optimize for (YYYY-MM-DD) or 'latest'")
    args = ap.parse_args()

    print("=== Medical Resource Agent Pipeline (Patched) ===")
    print(f"Raw timeseries: {RAW_TIMESERIES}")
    print(f"Geo costs:      {GEO_COSTS}\n")

    _ensure_dir(OUT_DIR)

    # 1) Forecast
    print("1) Forecasting…")
    fore = forecast_all()
    print(f"   → Saved {FORECAST_OUT} ({len(fore)} rows, horizon={FORECAST_HORIZON_DAYS}d)")

    # 2) Optimization for requested date (with slack)
    print("\n2) Optimization…")
    dist = load_distances()
    plan = optimize_transfers(fore, dist, target_date=args.date)
    print(f"   → Saved {ALLOC_OUT} ({len(plan)} rows) for date={args.date}")

    # 3) Rebuild FAISS
    print("\n3) Rebuilding FAISS index…")
    rebuild_faiss_index()

    # 4) Compact summary
    print("\n=== Summary ===")
    d0 = fore["forecast_date"].max() if args.date == "latest" else args.date
    s = (
        fore[fore["forecast_date"].astype(str) == str(d0)]
        .groupby("hospital_id", as_index=False)["urgency_score"]
        .mean()
        .sort_values("urgency_score", ascending=False)
    )
    print("Urgency (highest first) on", d0)
    print(s.to_string(index=False))

    if not plan.empty:
        grp = plan.groupby(["from", "to", "resource"], as_index=False)["quantity"].sum()
        print("\nTransfers:")
        for _, r in grp.iterrows():
            print(f"  {r['from']} → {r['to']} | {r['quantity']} {r['resource']}")
    else:
        print("\nNo transfers recommended by optimizer for this date (slack absorbed shortages).")

if __name__ == "__main__":
    # Make prints flush immediately
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
