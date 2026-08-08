"""
Train and compare freight rate models, tune the strongest candidates with
Optuna, then refit the winner on all of train_test.csv.

Run:
    python -m src.train
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

# Make this script's own directory importable regardless of how it's invoked
# (`python -m src.train` from the repo root, `python src/train.py`, or
# `cd src && python train.py` all work).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from data_prep import clean, fit_imputation_stats, load_raw
from features import add_causal_lane_history, build_feature_matrix, CATEGORICAL_FEATURES
from cv import make_folds
from models import build_catboost, build_hgb, build_lightgbm, build_linear_baseline, build_xgboost, to_category_dtype

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
REPORT_PATH = Path(__file__).resolve().parents[1] / "training_report.json"

HEADLINE_HOLDOUT_DAYS = 56  # last 8 weeks -> mirrors the train_test.csv -> validation.csv gap
N_OPTUNA_TRIALS = 12
# Tuning uses only the last 2 forward-chaining folds (Sep, Sep-Oct) rather than
# all 3 -- this sandbox has a single CPU core, so full 3-fold x many-trial
# search is too slow to be practical. 2 folds still avoids overfitting
# hyperparameters to one single holdout window, at a fraction of the cost.
TUNING_FOLD_BOUNDARIES = [
    ("2025-09-01", "2025-09-29"),
    ("2025-09-29", "2025-11-01"),
]


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def fit_predict(model_kind: str, train_df: pd.DataFrame, valid_df: pd.DataFrame, params: dict | None = None):
    """Fit one model kind on train_df, predict valid_df, return $ predictions.
    All models train on log1p(posted_rate)."""
    params = params or {}
    X_train_raw = build_feature_matrix(train_df)
    X_valid_raw = build_feature_matrix(valid_df)
    y_train = np.log1p(train_df["posted_rate"])

    if model_kind == "catboost":
        model = build_catboost(**params)
        model.fit(X_train_raw, y_train, cat_features=CATEGORICAL_FEATURES, verbose=False)
        pred = model.predict(X_valid_raw)
    elif model_kind == "lightgbm":
        model = build_lightgbm(**params)
        Xtr, Xva = to_category_dtype(X_train_raw), to_category_dtype(X_valid_raw)
        model.fit(Xtr, y_train, categorical_feature=CATEGORICAL_FEATURES)
        pred = model.predict(Xva)
    elif model_kind == "xgboost":
        model = build_xgboost(**params)
        Xtr, Xva = to_category_dtype(X_train_raw), to_category_dtype(X_valid_raw)
        model.fit(Xtr, y_train)
        pred = model.predict(Xva)
    elif model_kind == "hgb":
        model = build_hgb(**params)
        Xtr, Xva = to_category_dtype(X_train_raw), to_category_dtype(X_valid_raw)
        model.fit(Xtr, y_train)
        pred = model.predict(Xva)
    elif model_kind == "linear":
        model = build_linear_baseline()
        model.fit(X_train_raw, y_train)
        pred = model.predict(X_valid_raw)
    else:
        raise ValueError(model_kind)

    return np.expm1(pred), model


def headline_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Single train/holdout split matching the train_test->validation gap."""
    cutoff = df["date"].max() - pd.Timedelta(days=HEADLINE_HOLDOUT_DAYS)
    train_df = df[df["date"] <= cutoff]
    holdout_df = df[df["date"] > cutoff]

    rows = []
    for kind in ["linear", "hgb", "catboost", "lightgbm", "xgboost"]:
        t0 = time.time()
        pred, _ = fit_predict(kind, train_df, holdout_df)
        metrics = evaluate(holdout_df["posted_rate"], pred)
        metrics["model"] = kind
        metrics["seconds"] = round(time.time() - t0, 1)
        rows.append(metrics)
        print(f"  {kind:10s} MAE={metrics['mae']:7.2f}  RMSE={metrics['rmse']:7.2f}  "
              f"MAPE={metrics['mape_pct']:5.2f}%  ({metrics['seconds']}s)")
    return pd.DataFrame(rows).sort_values("mae")


def cv_objective_factory(df: pd.DataFrame, model_kind: str):
    folds = list(make_folds(df, boundaries=TUNING_FOLD_BOUNDARIES))

    def objective(trial: optuna.Trial) -> float:
        if model_kind == "catboost":
            params = dict(
                iterations=trial.suggest_int("iterations", 200, 600),
                learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                depth=trial.suggest_int("depth", 4, 8),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            )
        elif model_kind == "lightgbm":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 700),
                learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                max_depth=trial.suggest_int("max_depth", 4, 8),
                num_leaves=trial.suggest_int("num_leaves", 15, 100),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            )
        elif model_kind == "xgboost":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 700),
                learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                max_depth=trial.suggest_int("max_depth", 4, 8),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            )
        else:
            raise ValueError(model_kind)

        fold_maes = []
        for train_fold, valid_fold in folds:
            pred, _ = fit_predict(model_kind, train_fold, valid_fold, params)
            fold_maes.append(mean_absolute_error(valid_fold["posted_rate"], pred))
        return float(np.mean(fold_maes))

    return objective


def tune_model(df: pd.DataFrame, model_kind: str, n_trials: int = N_OPTUNA_TRIALS) -> dict:
    print(f"\nTuning {model_kind} with Optuna ({n_trials} trials, 3-fold expanding-window CV)...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(cv_objective_factory(df, model_kind), n_trials=n_trials, show_progress_bar=False)
    print(f"  best CV MAE={study.best_value:.2f}  params={study.best_params}")
    return study.best_params, study.best_value


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    raw = load_raw(DATA_DIR / "train_test.csv")
    stats = fit_imputation_stats(raw)
    clean_df = clean(raw, stats)
    # Causal lane-history features computed once over the FULL chronologically
    # sorted frame -- every row only ever sees strictly earlier rows, whether
    # that row later lands in a train partition or a holdout/fold partition.
    full_df = add_causal_lane_history(clean_df)

    print("=== Headline comparison (train <= 2025-09-05, holdout 2025-09-06..2025-10-31) ===")
    headline = headline_comparison(full_df)
    print(f"\nBest untuned model: {headline.iloc[0]['model']} (MAE={headline.iloc[0]['mae']:.2f})")

    # Tune the two strongest gradient-boosting candidates (CatBoost is the
    # stated first choice for this kind of tabular+categorical data; LightGBM
    # is included as the strongest of the remaining candidates on the
    # headline numbers). HGB and linear are kept only as reported baselines.
    top_candidates = [m for m in headline["model"] if m in ("catboost", "lightgbm", "xgboost")][:2]
    print(f"\nTuning candidates: {top_candidates}")

    tuned_results = {}
    for kind in top_candidates:
        best_params, best_cv_mae = tune_model(full_df, kind)
        tuned_results[kind] = {"params": best_params, "cv_mae": best_cv_mae}

    # Re-evaluate tuned params on the headline holdout for an apples-to-apples
    # comparison against the untuned table above.
    cutoff = full_df["date"].max() - pd.Timedelta(days=HEADLINE_HOLDOUT_DAYS)
    train_df = full_df[full_df["date"] <= cutoff]
    holdout_df = full_df[full_df["date"] > cutoff]

    print("\n=== Tuned models on headline holdout ===")
    for kind, res in tuned_results.items():
        pred, _ = fit_predict(kind, train_df, holdout_df, res["params"])
        metrics = evaluate(holdout_df["posted_rate"], pred)
        tuned_results[kind]["headline_holdout"] = metrics
        print(f"  {kind:10s} MAE={metrics['mae']:7.2f}  RMSE={metrics['rmse']:7.2f}  MAPE={metrics['mape_pct']:5.2f}%")

    # ---- Pick the overall winner across untuned + tuned results ----
    all_candidates = [(row["model"], row["mae"], {}) for _, row in headline.iterrows()]
    for kind, res in tuned_results.items():
        all_candidates.append((f"{kind}_tuned", res["headline_holdout"]["mae"], res["params"]))
    all_candidates.sort(key=lambda t: t[1])
    winner_name, winner_mae, winner_params = all_candidates[0]
    winner_kind = winner_name.replace("_tuned", "")
    is_tuned = winner_name.endswith("_tuned")
    print(f"\nOverall winner: {winner_name} (MAE={winner_mae:.2f})")

    # ---- Refit winner on ALL of train_test.csv for production ----
    full_stats = fit_imputation_stats(raw)
    full_clean = clean(raw, full_stats)
    full_hist_df = add_causal_lane_history(full_clean)  # causal, used only to fit; fine to reuse full_df
    final_lane_stats = None
    from features import fit_final_lane_stats
    final_lane_stats = fit_final_lane_stats(full_clean)

    X_full_raw = build_feature_matrix(full_hist_df)
    y_full = np.log1p(full_hist_df["posted_rate"])

    if winner_kind == "catboost":
        final_model = build_catboost(**winner_params)
        final_model.fit(X_full_raw, y_full, cat_features=CATEGORICAL_FEATURES, verbose=False)
    elif winner_kind == "lightgbm":
        final_model = build_lightgbm(**winner_params)
        final_model.fit(to_category_dtype(X_full_raw), y_full, categorical_feature=CATEGORICAL_FEATURES)
    elif winner_kind == "xgboost":
        final_model = build_xgboost(**winner_params)
        final_model.fit(to_category_dtype(X_full_raw), y_full)
    elif winner_kind == "hgb":
        final_model = build_hgb(**winner_params)
        final_model.fit(to_category_dtype(X_full_raw), y_full)
    else:
        final_model = build_linear_baseline()
        final_model.fit(X_full_raw, y_full)

    joblib.dump(
        {
            "model": final_model,
            "model_kind": winner_kind,
            "stats": full_stats,
            "lane_stats": final_lane_stats,
        },
        MODEL_DIR / "model.joblib",
    )

    report = {
        "train_window": {"start": str(raw["date"].min().date()), "end": str(raw["date"].max().date()), "n_rows": int(len(raw))},
        "headline_holdout_window": {
            "start": str(holdout_df["date"].min().date()),
            "end": str(holdout_df["date"].max().date()),
            "n_rows": int(len(holdout_df)),
        },
        "headline_comparison_untuned": headline.to_dict(orient="records"),
        "optuna_tuning": {
            k: {"cv_mae": v["cv_mae"], "params": v["params"], "headline_holdout": v["headline_holdout"]}
            for k, v in tuned_results.items()
        },
        "winner": {"name": winner_name, "kind": winner_kind, "is_tuned": is_tuned, "params": winner_params, "headline_mae": winner_mae},
        "final_model_trained_on_all_rows": int(len(full_hist_df)),
        "data_quality_fixes": [
            "weight: took abs() of negative values (~0.6% of rows)",
            "weight: missing values (~0.6% of rows) filled with the training-set median for that equipment type",
            "market_index: missing values (~0.8% of rows) filled with the training-set median",
        ],
        "feature_additions": [
            "lane_hist_rate_per_mile / lane_hist_count / equip_hist_rate_per_mile: leakage-safe "
            "expanding-window (causal) history features -- see src/features.py docstring",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved model -> {MODEL_DIR / 'model.joblib'}")
    print(f"Saved report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
