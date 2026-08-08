"""
Feature engineering shared by training and inference.

Lane-history features (NEW)
----------------------------
Prior lane pricing history is a strong signal for freight rates (a lane
that has historically priced high/low tends to keep doing so), but it is
easy to leak the label into this kind of feature. Two different, correct
computations are used depending on context:

1. CAUSAL / EXPANDING (used only inside train_test.csv, for model fitting
   and the internal time-based holdout eval): for each row, the lane's
   historical average rate_per_mile is computed using only rows for that
   same lane that occurred *strictly before* the current row's date
   (pandas `expanding().mean().shift(1)` on data pre-sorted by date). A
   row is never allowed to see its own value or any future value from its
   own lane. Lanes with no prior occurrence fall back to an equipment-level
   expanding average, then to a global expanding average, in that order.
   `lane_hist_count` (how many prior loads that lane has) is included too,
   so the model can learn to trust the lane average more when it's backed
   by more history.

2. STATIC / FINAL (used for validation.csv and december_chart_inputs.csv):
   these two files sit entirely *after* every date in train_test.csv, so
   there is no leakage risk in using each lane's full-history average
   computed across all of train_test.csv -- it all happened in the past
   relative to what we're predicting. Cities/lanes never seen in
   train_test.csv fall back the same way (equipment average -> global
   average).

Both paths share the same fallback logic (see `_coalesce_history`) so
train-time and inference-time features are computed identically in spirit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month",
    "dow",
    "doy",
    "week",
    "lane_hist_rate_per_mile",
    "lane_hist_count",
    "equip_hist_rate_per_mile",
]
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month
    out["dow"] = out["date"].dt.dayofweek
    out["doy"] = out["date"].dt.dayofyear
    # NOTE: deliberately NOT using pandas' ISO week (`.dt.isocalendar().week`).
    # ISO 8601 weeks wrap around the year boundary -- Dec 29-31 2025 fall in
    # "ISO week 1 of 2026", which the model would only ever have seen
    # associated with January dates during training (the cheapest month in
    # this data). That caused a spurious rate drop for the last few days of
    # the December chart that had nothing to do with real pricing. A simple
    # non-wrapping week-of-year (derived straight from day-of-year, no
    # year-boundary logic) avoids the artifact while keeping the same signal.
    out["week"] = ((out["doy"] - 1) // 7 + 1).astype(int)
    return out


def _coalesce_history(lane, equip, global_val, lane_count, min_lane_support: int = 2):
    """Prefer lane average once it has >= min_lane_support prior points,
    else fall back to the equipment-level average, else the global average.
    """
    use_lane = lane.notna() & (lane_count >= min_lane_support)
    result = np.where(use_lane, lane, equip)
    result = np.where(pd.isna(result), global_val, result)
    return result


def add_causal_lane_history(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe expanding-window lane history, for rows WITHIN
    train_test.csv only. `df` must contain `posted_rate` and `date`, and is
    returned sorted by date with the original index preserved as a column
    order (caller should re-sort back if needed)."""
    out = df.sort_values("date").copy()
    out["rate_per_mile"] = out["posted_rate"] / out["distance"]

    lane_key = out["pickup"] + " -> " + out["delivery"]
    out["_lane_key"] = lane_key

    lane_exp_mean = out.groupby("_lane_key")["rate_per_mile"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    lane_exp_count = out.groupby("_lane_key")["rate_per_mile"].transform(
        lambda s: s.expanding().count().shift(1)
    ).fillna(0)
    equip_exp_mean = out.groupby("equipment")["rate_per_mile"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    global_exp_mean = out["rate_per_mile"].expanding().mean().shift(1)
    # first-ever row has no history at all; fall back to the overall
    # dataset mean rather than leaving a NaN.
    global_exp_mean = global_exp_mean.fillna(out["rate_per_mile"].mean())
    equip_exp_mean = equip_exp_mean.fillna(global_exp_mean)

    out["lane_hist_rate_per_mile"] = _coalesce_history(
        lane_exp_mean, equip_exp_mean, global_exp_mean, lane_exp_count
    )
    out["lane_hist_count"] = lane_exp_count
    out["equip_hist_rate_per_mile"] = equip_exp_mean

    return out.drop(columns=["_lane_key", "rate_per_mile"])


def fit_final_lane_stats(train_df: pd.DataFrame) -> dict:
    """Full-history (non-causal) lookup tables computed across ALL of
    train_test.csv. Safe to use for validation.csv / december_chart_inputs.csv
    because both are entirely in the future relative to train_test.csv."""
    df = train_df.copy()
    df["rate_per_mile"] = df["posted_rate"] / df["distance"]
    lane_key = df["pickup"] + " -> " + df["delivery"]

    lane_stats = df.groupby(lane_key)["rate_per_mile"].agg(["mean", "count"])
    equip_stats = df.groupby("equipment")["rate_per_mile"].mean()
    global_mean = float(df["rate_per_mile"].mean())

    return {
        "lane_mean": lane_stats["mean"],
        "lane_count": lane_stats["count"],
        "equip_mean": equip_stats,
        "global_mean": global_mean,
    }


def apply_final_lane_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    lane_key = out["pickup"] + " -> " + out["delivery"]

    lane_mean = lane_key.map(stats["lane_mean"])
    lane_count = lane_key.map(stats["lane_count"]).fillna(0)
    equip_mean = out["equipment"].map(stats["equip_mean"]).fillna(stats["global_mean"])

    out["lane_hist_rate_per_mile"] = _coalesce_history(
        lane_mean, equip_mean, stats["global_mean"], lane_count
    )
    out["lane_hist_count"] = lane_count
    out["equip_hist_rate_per_mile"] = equip_mean
    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the model-ready feature columns, in a fixed order.
    Assumes calendar + lane-history features have already been added."""
    out = add_calendar_features(df)
    return out[FEATURE_COLUMNS].copy()
