import os
import re
import math
import uuid
import argparse
import pandas as pd
from datetime import datetime, timedelta

RAW_HHS = "data/external/hhs_hospital_capacity.csv"  # <-- put the Kaggle CSV here
OUT_TIMESERIES = "data/raw/hospital_timeseries.csv"

# Edit the list to the hospitals you want (must match 'hospital_name' in HHS CSV)
TARGET_HOSPITALS = [
    "Cleveland Clinic Main Campus",
    "Massachusetts General Hospital",
    "Cedars-Sinai Medical Center",
]

# How many days to build (rolling last N days from max CSV date)
DAYS = 30

def slug_id(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s[:24]  # keep it short-ish

def coalesce(*vals, default=0):
    for v in vals:
        if pd.notnull(v):
            return v
    return default

def require_columns(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in HHS CSV: {missing}")

def main():
    os.makedirs(os.path.dirname(OUT_TIMESERIES), exist_ok=True)
    if not os.path.exists(RAW_HHS):
        raise FileNotFoundError(f"Put the Kaggle file at: {RAW_HHS}")

    # Load HHS dataset
    df = pd.read_csv(RAW_HHS)

    # Common HHS columns (names can vary slightly by dump; these are typical)
    # If your dump uses different names, adjust here.
    # Typical useful fields:
    #   hospital_name, collection_date (or date), staffed_beds, inpatient_beds_used,
    #   icu_beds, adult_icu_beds_used, hospital_pk, latitude, longitude, city, state
    candidates = [
        "hospital_name", "collection_date", "date", "staffed_beds", "inpatient_beds_used",
        "all_adult_hospital_inpatient_beds", "all_adult_hospital_inpatient_beds_occupied",
        "icu_beds", "adult_icu_beds", "adult_icu_bed_covid_utilization_numerator",
        "adult_icu_beds_used", "adult_icu_beds_occupied", "hospital_pk", "ccn",
        "latitude", "longitude", "city", "state"
    ]
    have = [c for c in candidates if c in df.columns]
    if "hospital_name" not in df.columns:
        raise ValueError("HHS CSV must have 'hospital_name'")

    # Date column: prefer 'collection_date', else 'date'
    date_col = "collection_date" if "collection_date" in df.columns else ("date" if "date" in df.columns else None)
    if not date_col:
        raise ValueError("HHS CSV must have a date column: 'collection_date' or 'date'")

    # Standardize types
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)

    # Filter to hospitals of interest
    df = df[df["hospital_name"].isin(TARGET_HOSPITALS)].copy()
    if df.empty:
        raise ValueError(f"None of TARGET_HOSPITALS found in HHS CSV. Edit TARGET_HOSPITALS in this script.")

    # Pick a capacity mapping:
    # Beds capacity: prefer staffed_beds or all_adult_hospital_inpatient_beds
    df["beds_capacity"] = df.get("staffed_beds", pd.Series([None]*len(df)))
    if df["beds_capacity"].isna().all():
        df["beds_capacity"] = df.get("all_adult_hospital_inpatient_beds", 0)

    # Beds in use
    df["occupied_beds"] = df.get("inpatient_beds_used", pd.Series([None]*len(df)))
    if df["occupied_beds"].isna().all():
        df["occupied_beds"] = df.get("all_adult_hospital_inpatient_beds_occupied", 0)

    # ICU capacity
    df["icu_capacity"] = df.get("adult_icu_beds", pd.Series([None]*len(df)))
    if df["icu_capacity"].isna().all():
        df["icu_capacity"] = df.get("icu_beds", 0)

    # ICU used
    df["occupied_icu"] = df.get("adult_icu_beds_occupied", pd.Series([None]*len(df)))
    if df["occupied_icu"].isna().all():
        df["occupied_icu"] = df.get("adult_icu_beds_used", 0)

    # Ventilators: often missing; approximate from ICU capacity (e.g., 40% of ICU)
    df["ventilator_capacity"] = (df["icu_capacity"].fillna(0) * 0.4).round().astype(int)
    df["ventilators_in_use"] = (df["occupied_icu"].fillna(0) * 0.5).round().astype(int)

    # Aux fields not present in HHS: synthesize with reasonable defaults
    df["new_admissions"] = (df["occupied_beds"].fillna(0) * 0.06).round().astype(int)
    df["icu_admissions"] = (df["occupied_icu"].fillna(0) * 0.07).round().astype(int)
    df["avg_los_general"] = 4.5
    df["avg_los_icu"] = 6.5
    df["incoming_oxygen"] = (df["occupied_beds"].fillna(0) * 0.1).round().astype(int)
    df["incoming_portable_icu"] = 0
    df["notes"] = "hhs-derived"

    # Geo/admin
    df["region"] = df.get("state", "NA")
    df["latitude"] = pd.to_numeric(df.get("latitude", pd.Series([None]*len(df))), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", pd.Series([None]*len(df))), errors="coerce")

    # Keep only the last N days per hospital_name
    max_date = df[date_col].max()
    start_date = max_date - timedelta(days=DAYS - 1)
    df = df[(df[date_col] >= start_date) & (df[date_col] <= max_date)].copy()

    # Build unified schema
    out_rows = []
    for name, g in df.groupby("hospital_name"):
        hid = slug_id(name)
        for _, r in g.iterrows():
            beds_cap = int(coalesce(r.get("beds_capacity"), default=0))
            icu_cap = int(coalesce(r.get("icu_capacity"), default=0))
            occ_beds = int(coalesce(r.get("occupied_beds"), default=0))
            occ_icu = int(coalesce(r.get("occupied_icu"), default=0))
            vent_cap = int(coalesce(r.get("ventilator_capacity"), default=0))
            vents_use = int(coalesce(r.get("ventilators_in_use"), default=0))

            free_beds = max(beds_cap - occ_beds, 0)
            free_icu = max(icu_cap - occ_icu, 0)
            util = 0.0
            if beds_cap > 0:
                util = round(occ_beds / beds_cap, 3)

            out_rows.append({
                "date": r[date_col].date().isoformat(),
                "hospital_id": hid,
                "name": name,
                "region": r.get("region", "NA"),
                "beds_capacity": beds_cap,
                "icu_capacity": icu_cap,
                "ventilator_capacity": vent_cap,
                "occupied_beds": occ_beds,
                "occupied_icu": occ_icu,
                "ventilators_in_use": vents_use,
                "new_admissions": int(r.get("new_admissions", 0)),
                "icu_admissions": int(r.get("icu_admissions", 0)),
                "avg_los_general": float(r.get("avg_los_general", 4.5)),
                "avg_los_icu": float(r.get("avg_los_icu", 6.5)),
                "incoming_oxygen": int(r.get("incoming_oxygen", 0)),
                "incoming_portable_icu": int(r.get("incoming_portable_icu", 0)),
                "notes": "hhs-derived",
                "free_beds": int(free_beds),
                "free_icu": int(free_icu),
                "utilization_rate": util,
                "lat": float(r.get("latitude")) if pd.notnull(r.get("latitude")) else None,
                "lon": float(r.get("longitude")) if pd.notnull(r.get("longitude")) else None,
            })

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise RuntimeError("No rows generated. Check TARGET_HOSPITALS or CSV columns.")
    out.to_csv(OUT_TIMESERIES, index=False)
    print(f"✅ Wrote {len(out)} rows → {OUT_TIMESERIES}")

if __name__ == "__main__":
    main()
