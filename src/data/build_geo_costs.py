import os
import math
import pandas as pd

TIMESERIES = "data/raw/hospital_timeseries.csv"
OUT = "data/processed/geo_costs.csv"

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    to_rad = math.pi/180
    dlat = (lat2 - lat1) * to_rad
    dlon = (lon2 - lon1) * to_rad
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1*to_rad) * math.cos(lat2*to_rad) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_csv(TIMESERIES)
    hospitals = df[["hospital_id", "name", "lat", "lon"]].drop_duplicates()
    hospitals = hospitals.dropna(subset=["lat", "lon"])

    rows = []
    for i, a in hospitals.iterrows():
        for j, b in hospitals.iterrows():
            if a["hospital_id"] == b["hospital_id"]:
                continue
            km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            rows.append({
                "from": a["hospital_id"],
                "to": b["hospital_id"],
                "distance_km": round(km, 1),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"✅ Wrote {len(out)} pairs → {OUT}")

if __name__ == "__main__":
    main()
