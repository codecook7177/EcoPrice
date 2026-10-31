"""Markdown schedule used by the EcoPrice dashboard.

The dashboard shows discounts from this rule-based schedule. Its base term
follows the trained PPO policy's mean discount by effective days to expiry,
with additions for excess stock and for hot or humid weather. The same
function is called by src/store_state.py and by models/ppo_agent.py, so the
schedule evaluated in the simulator is the one shown to store managers.
"""

import numpy as np

CATEGORY_CLIMATE_VULNERABILITY = {
    "Meat & Fish": 0.12,
    "Dairy": 0.09,
    "Dairy & Eggs": 0.09,
    "Fruits": 0.07,
    "Vegetables": 0.06,
    "Bakery": 0.04,
    "Snacks": 0.03,
    "Specialty & Fresh": 0.06,
}


def schedule_discount(category, effective_days_float, effective_days, pred_demand, stock,
                      spoilage_mult=1.0, max_disc_cap=0.30):
    """Return (discount fraction, coverage ratio) for one batch."""
    cat = category
    d_eff = max(0.0, float(effective_days_float))
    c_rem = pred_demand * max(1.0, d_eff)
    coverage_ratio = stock / max(1.0, c_rem)

    # base discount by effective days to expiry
    if d_eff <= 0.5:
        if cat in ["Meat & Fish", "Meat_and_Fish"]:
            urg_base = 0.28
        elif cat in ["Bakery", "Dairy"]:
            urg_base = 0.22
        else:
            urg_base = 0.20
    elif d_eff <= 1.4 or effective_days <= 1:
        if cat in ["Meat & Fish", "Meat_and_Fish"]:
            urg_base = 0.28
        elif cat in ["Bakery", "Dairy", "Vegetables"]:
            urg_base = 0.18
        else:
            urg_base = 0.16
    elif d_eff <= 2.2:
        urg_base = 0.06 * np.exp(-0.7 * (d_eff - 1.2))
    elif d_eff <= 4.5:
        urg_base = 0.04 * np.exp(-0.8 * (d_eff - 2.2))
    else:
        urg_base = 0.0

    # stock in excess of expected sales before expiry
    if coverage_ratio > 1.0:
        excess_ratio = min(3.0, coverage_ratio - 1.0)
        time_urgency = 1.0 / max(0.8, d_eff)
        overhang_lift = 0.035 * excess_ratio * min(2.0, time_urgency)
    else:
        overhang_lift = 0.0

    # hot or humid weather
    vuln = CATEGORY_CLIMATE_VULNERABILITY.get(cat, 0.06)
    climate_lift = max(0.0, spoilage_mult - 1.0) * vuln * 0.35 if d_eff <= 4.0 else 0.0

    total = urg_base + overhang_lift + climate_lift

    # no markdown when fresh stock is expected to sell at full price
    if d_eff >= 4.5 and coverage_ratio <= 1.2:
        total = 0.0
    elif d_eff >= 2.5 and coverage_ratio <= 1.0:
        total = 0.0

    return float(np.clip(total, 0.0, max_disc_cap)), coverage_ratio
