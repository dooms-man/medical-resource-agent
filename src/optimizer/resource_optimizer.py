"""
resource_optimizer.py  —  Multi-resource LP for hospital resource redistribution

Features:
- Optimizes ICU beds, general beds, and ventilators.
- Minimizes urgency-weighted unmet needs + distance-based transfer costs.
- Reads hospital capacities, forecasted surpluses/shortages, and geo distances.
- Outputs: allocation_plan.csv + allocation_summary.json
"""

import pandas as pd
from pathlib import Path
import argparse
import pulp
import json


# ============================
# Data Loading Functions
# ============================
def load_data(processed_dir: str):
    processed_dir = Path(processed_dir)
    df_fore = pd.read_csv(processed_dir / "forecast_alerts.csv", parse_dates=["forecast_date"])
    df_base = pd.read_csv(processed_dir / "hospital_timeseries.csv", parse_dates=["date"])
    df_geo = pd.read_csv(processed_dir / "geo_costs.csv")
    return df_fore, df_base, df_geo


# ============================
# Data Preparation
# ============================
def prepare_supply_demand(df_fore, df_base):
    # Latest snapshot of hospital capacities
    latest = df_base.sort_values(["hospital_id", "date"]).groupby("hospital_id").last().reset_index()

    chosen_date = df_fore["forecast_date"].min()
    df_focus = df_fore[df_fore["forecast_date"] == chosen_date]

    hospitals = sorted(df_focus["hospital_id"].unique().tolist())

    resources = ["icu", "beds", "ventilators"]
    surplus = {r: {} for r in resources}
    shortage = {r: {} for r in resources}
    urgency = {}
    capacity = {r: {} for r in resources}

    for _, row in df_focus.iterrows():
        hid = row["hospital_id"]
        urg = float(row.get("urgency_score", 0.0))
        urgency[hid] = urg

        # Map capacities
        cap_row = latest[latest["hospital_id"] == hid]
        if not cap_row.empty:
            capacity["icu"][hid] = int(cap_row.iloc[0].get("icu_capacity", 0))
            capacity["beds"][hid] = int(cap_row.iloc[0].get("beds_capacity", 0))
            capacity["ventilators"][hid] = int(cap_row.iloc[0].get("ventilator_capacity", 0))
        else:
            capacity["icu"][hid] = capacity["beds"][hid] = capacity["ventilators"][hid] = 0

        # Surplus/shortage calculation
        for res, col in zip(resources, ["pred_free_icu", "pred_free_beds", "pred_free_ventilators"]):
            val = float(row.get(col, 0.0))
            if val >= 0:
                surplus[res][hid] = int(round(val))
                shortage[res][hid] = 0
            else:
                surplus[res][hid] = 0
                shortage[res][hid] = int(round(-val))

    return hospitals, resources, surplus, shortage, urgency, capacity, chosen_date


# ============================
# Build Cost Matrix
# ============================
def build_cost_matrix(df_geo):
    cost_matrix = {}
    for _, row in df_geo.iterrows():
        i, j, dist = row["from_hospital"], row["to_hospital"], float(row["distance_km"])
        if i not in cost_matrix:
            cost_matrix[i] = {}
        cost_matrix[i][j] = dist
    return cost_matrix


# ============================
# LP Model
# ============================
def build_and_solve(hospitals, resources, surplus, shortage, urgency, capacity, cost_matrix,
                    integral=True, transfer_cost_weight=0.02):
    prob = pulp.LpProblem("multi_resource_allocation", pulp.LpMinimize)

    # Decision variables
    x = {}  # transfer vars
    unmet = {}  # unmet vars

    for res in resources:
        for i in hospitals:
            for j in hospitals:
                if i == j:
                    continue
                var = pulp.LpVariable(f"x_{res}_{i}_to_{j}", lowBound=0, cat="Integer" if integral else "Continuous")
                x[(res, i, j)] = var

        for j in hospitals:
            unmet[(res, j)] = pulp.LpVariable(f"unmet_{res}_{j}", lowBound=0, cat="Integer" if integral else "Continuous")

    # Objective: minimize urgency-weighted unmet + distance-weighted transfer cost
    obj_unmet = pulp.lpSum([urgency[j] * unmet[(res, j)] for res in resources for j in hospitals])
    obj_transfer = pulp.lpSum([
        cost_matrix.get(i, {}).get(j, 50.0) * x[(res, i, j)]
        for res in resources for i in hospitals for j in hospitals if (res, i, j) in x
    ])

    prob += obj_unmet + transfer_cost_weight * obj_transfer, "Total_Cost"

    # Constraints
    for res in resources:
        # Supply: cannot send more than surplus
        for i in hospitals:
            outbound = [x[(res, i, j)] for j in hospitals if (res, i, j) in x]
            if outbound:
                prob += pulp.lpSum(outbound) <= surplus[res].get(i, 0), f"supply_{res}_{i}"

        # Demand: cannot receive more than shortage
        for j in hospitals:
            inbound = [x[(res, i, j)] for i in hospitals if (res, i, j) in x]
            if inbound:
                prob += pulp.lpSum(inbound) <= shortage[res].get(j, 0), f"demand_{res}_{j}"

        # Unmet shortage definition
        for j in hospitals:
            received = pulp.lpSum([x[(res, i, j)] for i in hospitals if (res, i, j) in x])
            prob += unmet[(res, j)] >= shortage[res].get(j, 0) - received, f"unmet_{res}_{j}"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    # Collect results
    allocations = []
    for (res, i, j), var in x.items():
        qty = int(round(var.value() or 0))
        if qty > 0:
            allocations.append({
                "from": i, "to": j,
                "resource": res,
                "quantity": qty,
                "distance_km": cost_matrix.get(i, {}).get(j, None)
            })

    unmet_after = {(res, j): int(round(unmet[(res, j)].value() or 0))
                   for res in resources for j in hospitals}

    result = {
        "status": status,
        "allocations": allocations,
        "unmet_after": unmet_after,
        "objective_value": pulp.value(prob.objective)
    }

    return result


# ============================
# Save Results
# ============================
def save_plan(out_dir: str, chosen_date, result):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_plan = pd.DataFrame(result["allocations"])
    df_plan.insert(0, "date", chosen_date)
    plan_path = out_dir / "allocation_plan.csv"
    df_plan.to_csv(plan_path, index=False)

    summary = {
        "date": str(chosen_date),
        "status": result["status"],
        "objective": result["objective_value"],
        "total_transferred": int(df_plan["quantity"].sum()) if not df_plan.empty else 0,
        "num_transfers": len(df_plan),
    }

    with open(out_dir / "allocation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved plan → {plan_path}")
    print(f"✅ Saved summary → {out_dir/'allocation_summary.json'}")


# ============================
# Main Entrypoint
# ============================
def main(processed_dir="data/processed", out_dir="data/processed",
         integral=True, transfer_cost_weight=0.02):
    df_fore, df_base, df_geo = load_data(processed_dir)
    hospitals, resources, surplus, shortage, urgency, capacity, chosen_date = prepare_supply_demand(df_fore, df_base)
    cost_matrix = build_cost_matrix(df_geo)

    print("\n=== Optimization Summary ===")
    print("Hospitals:", hospitals)
    print("Resources:", resources)
    print("Transfer cost weight:", transfer_cost_weight)
    print("Solving LP...\n")

    result = build_and_solve(hospitals, resources, surplus, shortage,
                             urgency, capacity, cost_matrix,
                             integral=integral, transfer_cost_weight=transfer_cost_weight)

    print("Status:", result["status"])
    print("Objective Value:", result["objective_value"])
    print("Total Transfers:", len(result["allocations"]))
    for a in result["allocations"]:
        print(f"  {a['from']} → {a['to']} | {a['quantity']} {a['resource']} (dist={a['distance_km']} km)")

    save_plan(out_dir, chosen_date, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--integral", type=bool, default=True)
    parser.add_argument("--transfer_cost_weight", type=float, default=0.02)
    args = parser.parse_args()

    main(args.processed_dir, args.out_dir, args.integral, args.transfer_cost_weight)
