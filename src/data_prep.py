"""
Data loading and cleaning.

Issues found during EDA and how they're handled:
  1. `weight` has ~0.6% negative values in both train_test.csv and
     validation.csv. distance/market_index/quote_signal never go negative,
     and the absolute-value distribution of the negative rows is identical
     to the positive population -> these are sign-entry errors, not a
     distinct population. Fixed with abs(weight).
  2. `weight` and `market_index` have a small number of missing values.
     Imputed with the TRAINING-split median (weight: per equipment type,
     since Reefer/Flatbed/Dry Van loads have structurally different typical
     weights; market_index: single training-split median), so no
     information from a held-out window leaks into the imputer.
  3. `date` is parsed to a real datetime for calendar + lane-history
     feature engineering.
  4. pickup/delivery/equipment are kept as raw strings/category dtype so
     each model's native categorical handling can be used (CatBoost takes
     raw strings; LightGBM/XGBoost/HGB take pandas 'category' dtype).
     validation.csv has 8 pickup/delivery cities that never appear in
     train_test.csv, so city name alone is not fully reliable -- lat/lon
     and lane-history features (features.py) provide numeric fallback
     signal for those lanes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CATEGORICAL_COLS = ["pickup", "delivery", "equipment"]


@dataclass
class ImputationStats:
    """Values learned from a training split only -- never from data being predicted."""
    weight_median_by_equipment: pd.Series
    weight_median_global: float
    market_index_median: float


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fit_imputation_stats(train_df: pd.DataFrame) -> ImputationStats:
    cleaned_weight = train_df["weight"].abs()
    return ImputationStats(
        weight_median_by_equipment=cleaned_weight.groupby(train_df["equipment"]).median(),
        weight_median_global=float(cleaned_weight.median()),
        market_index_median=float(train_df["market_index"].median()),
    )


def clean(df: pd.DataFrame, stats: ImputationStats) -> pd.DataFrame:
    out = df.copy()

    # Rule 1: negative weight -> sign error -> absolute value.
    out["weight"] = out["weight"].abs()

    # Rule 2: missing weight -> median weight for that equipment type,
    # falling back to the global training median if the equipment type
    # itself is somehow absent from the training stats.
    per_equipment_median = out["equipment"].map(stats.weight_median_by_equipment)
    out["weight"] = out["weight"].fillna(per_equipment_median).fillna(stats.weight_median_global)

    # Rule 3: missing market_index -> training-set median.
    if "market_index" in out.columns:
        out["market_index"] = out["market_index"].fillna(stats.market_index_median)

    return out
