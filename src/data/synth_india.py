import os, random
from datetime import date, timedelta
import numpy as np
import pandas as pd

OUT = "data/processed/hospital_timeseries.csv"
N_DAYS = 30
SEED = 42

# 10 Indian-style hospital names (feel free to tweak later)
HOSPITALS = [
    ("pune-sanjivani-hospital", "Sanjivani Hospital", "Pune"),
    ("mumbai-lotus-care", "Lotus Care Center", "Mumbai"),
    ("delhi-aarogyam", "Aarogyam Multi-Speciality", "Delhi"),
    ("nagpur-shree-krishna", "Shree Krishna Hospital", "Nagpur"),
    ("kolkata-sevamrut", "Sevamrut Medical Institute", "Kolkata"),
    ("ahmedabad-sahyog", "Sahyog Hospital", "Ahmedabad"),
    ("chennai-maa-janani", "Maa Janani Healthcare", "Chennai"),
    ("hyderabad-sparsh", "Sparsh Super Specialty", "Hyderabad"),
    ("jaipur-royal-medicity", "Royal Medicity", "Jaipur"),
    ("bengaluru-nirmal-ayu", "Nirmal Ayu Hospital", "Bengaluru"),
]

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    random.seed(SEED); np.random.seed(SEED)
    today = date.today()
    start = today - timedelta(days=N_DAYS-1)

    rows = []
    for hid, name, city in HOSPITALS:
        # Base capacities per hospital
        total_beds = random.randint(150, 400)
        total_icu  = random.randint(12, 40)

        # Daily variation drivers
        occ_beds = int(total_beds * np.random.uniform(0.60, 0.98))
        occ_icu  = int(total_icu * np.random.uniform(0.35, 0.90))

        # Staff baseline: approx 1 nurse per 5 beds + ICU emphasis
        staff_required_base = int(total_beds/5 + total_icu*1.5)

        for d in range(N_DAYS):
            dt = start + timedelta(days=d)

            # Admissions/discharges as random walk around occupancy
            adm = max(0, int(np.random.normal(loc=total_beds*0.05, scale=3)))
            dis = max(0, int(np.random.normal(loc=total_beds*0.048, scale=3)))

            # Update occupancy with noise
            occ_beds = clamp(occ_beds + adm - dis + int(np.random.normal(0, 3)), 0, total_beds)
            icu_adm = max(0, int(np.random.normal(loc=total_icu*0.05, scale=1)))
            icu_dis = max(0, int(np.random.normal(loc=total_icu*0.046, scale=1)))
            occ_icu  = clamp(occ_icu + icu_adm - icu_dis + int(np.random.normal(0, 1)), 0, total_icu)

            staff_required  = clamp(int(np.random.normal(staff_required_base, 2)), 1, 9999)
            # Staff available usually a bit short in stress
            staff_available = clamp(int(staff_required * np.random.uniform(0.88, 1.02)), 0, 9999)

            free_beds = total_beds - occ_beds
            free_icu  = total_icu - occ_icu
            util_rate = round(occ_beds / total_beds, 3) if total_beds else 0.0

            rows.append({
                "date": dt.isoformat(),
                "hospital_id": hid,
                "name": name,
                "region": city,
                "total_beds": total_beds,
                "total_icu": total_icu,
                "occupied_beds": occ_beds,
                "occupied_icu": occ_icu,
                "admissions": adm,
                "discharges": dis,
                "staff_required": staff_required,
                "staff_available": staff_available,
                "free_beds": free_beds,
                "free_icu": free_icu,
                "utilization_rate": util_rate,
                "notes": "synthetic-india-30d"
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"✅ Wrote {len(df)} rows → {OUT}")

if __name__ == "__main__":
    main()
