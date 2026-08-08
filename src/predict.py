"""
Generate the two required prediction outputs using the saved winning model:
  1. validation_predictions.csv         (load_id, predicted_rate, 12,000 rows)
  2. data/december_chart_inputs.csv     (predicted_rate filled, 31 rows)

Feature gaps in each file, and how they're filled:
  - validation.csv provides every column the model was trained on except
    the two lane-history features, which are computed here using the
    STATIC/full-history lane stats saved at training time (safe: all of
    train_test.csv precedes validation.csv in time -- see features.py).
  - december_chart_inputs.csv additionally lacks lat/lon and
    market_index/quote_signal:
      * lat/lon: looked up from train_test.csv (Lexington and Fort Wayne
        both appear there with one fixed coordinate pair each).
      * market_index/quote_signal: no lane-specific or date-specific
        source exists for future December values, so they're set to the
        median of the most recent 30 days of train_test.csv (2025-10-02
        to 2025-10-31) as a flat "current conditions" proxy. This is the
        single biggest simplification in this pipeline -- see the report.

Usage
-----
    python -m src.predict
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make this script's own directory importable regardless of how it's invoked
# (`python -m src.predict` from the repo root, `python src/predict.py`, or
# `cd src && python predict.py` all work).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd

from data_prep import clean, load_raw
from features import apply_final_lane_stats, build_feature_matrix
from models import to_category_dtype

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
OUT_DIR = Path(__file__).resolve().parents[1]


def city_lookup(train_df: pd.DataFrame) -> pd.DataFrame:
    pickup_map = train_df[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates()
    pickup_map.columns = ["city", "lat", "lon"]
    delivery_map = train_df[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates()
    delivery_map.columns = ["city", "lat", "lon"]
    return pd.concat([pickup_map, delivery_map]).drop_duplicates(subset="city").set_index("city")


def model_predict(bundle: dict, X_raw: pd.DataFrame) -> np.ndarray:
    kind = bundle["model_kind"]
    model = bundle["model"]
    if kind == "catboost":
        pred = model.predict(X_raw)
    else:
        pred = model.predict(to_category_dtype(X_raw))
    return np.expm1(pred)


def predict_validation(bundle: dict, stats, lane_stats, train_raw: pd.DataFrame) -> None:
    val_raw = load_raw(DATA_DIR / "validation.csv")
    val_clean = clean(val_raw, stats)
    val_clean = apply_final_lane_stats(val_clean, lane_stats)
    X_val = build_feature_matrix(val_clean)

    predicted = model_predict(bundle, X_val)
    predicted = np.clip(predicted, 1.0, None)

    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    out = template[["load_id"]].merge(
        pd.DataFrame({"load_id": val_raw["load_id"], "predicted_rate": np.round(predicted, 2)}),
        on="load_id",
        how="left",
    )
    assert out["predicted_rate"].notna().all(), "missing predictions for some load_id"
    out_path = OUT_DIR / "validation_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} rows -> {out_path}")


def predict_december(bundle: dict, stats, lane_stats, train_raw: pd.DataFrame) -> None:
    dec = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")
    dec["date"] = pd.to_datetime(dec["date"])

    lookup = city_lookup(train_raw)
    dec["pickup_lat"] = lookup.loc[dec["pickup"], "lat"].values
    dec["pickup_lon"] = lookup.loc[dec["pickup"], "lon"].values
    dec["delivery_lat"] = lookup.loc[dec["delivery"], "lat"].values
    dec["delivery_lon"] = lookup.loc[dec["delivery"], "lon"].values

    recent = train_raw[train_raw["date"] > train_raw["date"].max() - pd.Timedelta(days=30)]
    dec["market_index"] = float(recent["market_index"].median())
    dec["quote_signal"] = float(recent["quote_signal"].median())

    dec_clean = clean(dec, stats)
    dec_clean = apply_final_lane_stats(dec_clean, lane_stats)
    X_dec = build_feature_matrix(dec_clean)

    predicted = model_predict(bundle, X_dec)
    predicted = np.clip(predicted, 1.0, None)
    dec["predicted_rate"] = np.round(predicted, 2)

    out_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    out = dec[out_cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out_path = DATA_DIR / "december_chart_inputs.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")
    print(out[["date", "predicted_rate"]].to_string(index=False))


def main() -> None:
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    stats = bundle["stats"]
    lane_stats = bundle["lane_stats"]
    train_raw = load_raw(DATA_DIR / "train_test.csv")

    predict_validation(bundle, stats, lane_stats, train_raw)
    predict_december(bundle, stats, lane_stats, train_raw)


if __name__ == "__main__":
    main()
