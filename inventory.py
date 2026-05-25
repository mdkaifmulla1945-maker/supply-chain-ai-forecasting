import numpy as np

def compute_inventory_metrics(prediction, current_stock, lead_time=2):

    # predicted demand over next 3 weeks
    demand_3w = sum(prediction)

    avg_weekly_demand = np.mean(prediction)

    # reorder point (simple industry formula)
    reorder_point = avg_weekly_demand * lead_time

    # stockout risk
    risk = max(0, (demand_3w - current_stock) / (demand_3w + 1))

    status = "SAFE"

    if current_stock < reorder_point:
        status = "REORDER"
    if current_stock < avg_weekly_demand:
        status = "CRITICAL"

    return {
        "predicted_demand_3w": float(demand_3w),
        "reorder_point": float(reorder_point),
        "stockout_risk": float(risk),
        "status": status
    }