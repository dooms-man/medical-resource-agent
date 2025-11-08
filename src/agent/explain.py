"""
explain.py
----------
Explainability module for the Medical Resource Allocation Agent.

Reads processed artifacts:
- data/processed/forecast_alerts.csv
- data/processed/allocation_plan.csv
- (optional) data/processed/faiss_index/  -> for retrieval trace

Outputs a human-readable explanation + evidence used for a given (or latest) forecast date.

Run examples:
  python src/agent/explain.py
  python src/agent/explain.py --date 2025-11-07
  python src/agent/explain.py --trace 5
"""

import os
import argparse
import pandas as pd
from typing import Optional, Tuple, Dict

PROCESSED_DIR = "data/processed"
FORECAST_CSV = os.path.join(PROCESSED_DIR, "forecast_alerts.csv")
PLAN_CSV     = os.path.join(PROCESSED_DIR, "allocation_plan.csv")
FAISS_DIR    = os.path.join(PROCESSED_DIR, "faiss_index")


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


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return _clean_cols(pd.read_csv(path))


def pick_date(fore: pd.DataFrame, date: Optional[str]) -> str:
    """Pick a target forecast_date (user-provided or latest available)."""
    if date:
        if date not in set(fore["forecast_date"].astype(str).unique()):
            raise ValueError(f"Requested date {date} not found in forecast_alerts.csv")
        return date
    # default: latest (max)
    d = str(pd.to_datetime(fore["forecast_date"]).max().date())
    return d


def build_supply_demand_for_date(fore: pd.DataFrame, d0: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    day = fore[fore["forecast_date"].astype(str) == d0].copy()
    # Ensure needed columns exist
    for col in ["hospital_id", "pred_free_beds", "pred_free_icu", "urgency_score"]:
        if col not in day.columns:
            raise ValueError(f"forecast_alerts.csv missing column: {col}")

    # Supply = positive free; Demand = positive shortage (negative free)
    day["beds_supply"]  = day["pred_free_beds"].clip(lower=0)
    day["icu_supply"]   = day["pred_free_icu"].clip(lower=0)
    day["beds_demand"]  = (-day["pred_free_beds"]).clip(lower=0)
    day["icu_demand"]   = (-day["pred_free_icu"]).clip(lower=0)

    supply = day[["hospital_id", "beds_supply", "icu_supply"]].copy()
    demand = day[["hospital_id", "beds_demand", "icu_demand"]].copy()
    return supply, demand


def load_plan_for_date(plan: pd.DataFrame, d0: str) -> pd.DataFrame:
    # Some plans may store date as datetime/str; normalize to str
    plan = plan.copy()
    if "date" in plan.columns:
        plan["date"] = plan["date"].astype(str)
        plan_d = plan[plan["date"] == d0].copy()
    else:
        plan_d = plan.copy()
    # normalize schema variations
    for needed in ["from", "to", "resource", "quantity", "distance_km"]:
        if needed not in plan_d.columns:
            # try to recover common variants
            if needed == "from":
                for c in plan_d.columns:
                    if c in ("source", "sender", "from_hospital", "hospital_from"):
                        plan_d = plan_d.rename(columns={c: "from"})
            elif needed == "to":
                for c in plan_d.columns:
                    if c in ("destination", "receiver", "to_hospital", "hospital_to"):
                        plan_d = plan_d.rename(columns={c: "to"})
            elif needed == "distance_km":
                for c in plan_d.columns:
                    if c in ("distance", "km", "dist_km"):
                        plan_d = plan_d.rename(columns={c: "distance_km"})
    return plan_d


def explain(date: Optional[str] = None, trace_k: int = 0) -> str:
    fore = _read_csv(FORECAST_CSV)
    plan = _read_csv(PLAN_CSV) if os.path.exists(PLAN_CSV) else pd.DataFrame(columns=["from","to","resource","quantity","distance_km"])

    d0 = pick_date(fore, date)

    # 1) Urgency ranking for the target date
    day = _clean_cols(fore[fore["forecast_date"].astype(str) == d0].copy())
    if day.empty:
        raise ValueError(f"No forecast rows found for {d0}")

    urg_rank = (
        day[["hospital_id","urgency_score","pred_free_beds","pred_free_icu"]]
        .sort_values("urgency_score", ascending=False)
        .reset_index(drop=True)
    )

    top_hosp = urg_rank.iloc[0]["hospital_id"]
    top_urg  = float(urg_rank.iloc[0]["urgency_score"])

    # 2) Evidence rows for the most urgent hospital
    evidence = day[day["hospital_id"] == top_hosp][[
        "hospital_id","forecast_date","pred_free_beds","pred_free_icu","urgency_score"
    ]].copy()

    # 3) Supply/Demand summary used by optimizer
    supply, demand = build_supply_demand_for_date(fore, d0)

    # 4) Transfer plan for the date
    plan_d = load_plan_for_date(plan, d0)
    plan_grp = pd.DataFrame()
    if not plan_d.empty:
        plan_grp = (
            plan_d.groupby(["from","to","resource"], as_index=False)["quantity"]
                  .sum()
                  .sort_values(["to","resource"], ascending=[True, True])
        )

    # --- Build human-readable explanation ---
    lines = []
    lines.append(f"=== Explainability Report for {d0} ===")
    lines.append("")
    lines.append("Top Urgency (highest first):")
    lines.append(urg_rank.to_string(index=False))
    lines.append("")

    # Reason for top urgency
    top_row = evidence.iloc[0]
    reason_parts = []
    if float(top_row["pred_free_icu"]) < 0:
        reason_parts.append(f"ICU shortage ({top_row['pred_free_icu']})")
    if float(top_row["pred_free_beds"]) < 0:
        reason_parts.append(f"Beds shortage ({top_row['pred_free_beds']})")
    if not reason_parts:
        reason_parts.append("High risk but non-negative free values (likely near-capacity ICU/beds)")

    lines.append(f"Primary focus hospital: {top_hosp} (urgency={top_urg:.3f})")
    lines.append("Reason: " + "; ".join(reason_parts))
    lines.append("")

    lines.append("Evidence rows (from forecast_alerts.csv):")
    lines.append(evidence.to_string(index=False))
    lines.append("")

    # Supply / Demand
    lines.append("Supply (positive free capacity):")
    lines.append(supply.to_string(index=False))
    lines.append("")
    lines.append("Demand (shortage magnitude):")
    lines.append(demand.to_string(index=False))
    lines.append("")

    if plan_grp.empty:
        lines.append("No transfers planned by optimizer for this date.")
    else:
        lines.append("Planned Transfers (aggregated):")
        lines.append(plan_grp.to_string(index=False))
        lines.append("")
        # Short, natural-language justification per receiving hospital
        for to_h in plan_grp["to"].unique():
            recv = plan_grp[plan_grp["to"] == to_h]
            recv_lines = []
            for _, r in recv.iterrows():
                recv_lines.append(f"{r['from']} → {r['to']} | {r['quantity']} {r['resource']}")
            lines.append(f"Why {to_h}? Because its forecasted free ICU/beds are low/negative (see evidence), and the transfers above reduce that gap while accounting for distance costs.")
            lines.append(" • " + " ; ".join(recv_lines))
        lines.append("")

    # Optional FAISS trace: show top K chunks most likely used by RAG (best-effort)
    if trace_k > 0:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vs = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
                retriever = vs.as_retriever(search_kwargs={"k": trace_k})
                # A canonical query aligned with the date
                q = f"Summarize forecasted shortages and planned transfers for {d0}."
                docs = retriever.invoke(q)
                lines.append(f"Retrieval Trace (top {trace_k} chunks):")
                for i, d in enumerate(docs, 1):
                    # page_content only; metadata may not be present depending on your builder
                    snip = d.page_content.replace("\n", " ")
                    if len(snip) > 220:
                        snip = snip[:220] + "…"
                    lines.append(f"{i}. {snip}")
            else:
                lines.append("Retrieval Trace: FAISS index not found; run the pipeline to rebuild it.")
        except Exception as e:
            lines.append(f"Retrieval Trace unavailable ({e}).")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="Target forecast_date (YYYY-MM-DD). If omitted, uses latest.")
    ap.add_argument("--trace", type=int, default=0, help="Show top-K retrieved chunks from FAISS (0 to disable).")
    args = ap.parse_args()

    report = explain(date=args.date, trace_k=args.trace)
    print(report)


if __name__ == "__main__":
    main()
