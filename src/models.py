"""
Model builders. Each candidate handles categorical features (pickup,
delivery, equipment) using its own native mechanism rather than one-hot
encoding, so unseen categories at inference (8 validation-only cities)
degrade gracefully instead of erroring out or silently vanishing:

  - CatBoost:    raw string columns, passed via `cat_features` indices.
  - LightGBM:    pandas 'category' dtype columns.
  - XGBoost:     pandas 'category' dtype columns (`enable_categorical=True`).
  - HGB (sklearn): pandas 'category' dtype columns (`categorical_features="from_dtype"`).
  - Linear:      one-hot (baseline only; included for comparison, not a
                  serious candidate for this feature set).

All models are trained on log1p(posted_rate) (see train.py for why) and
predict in dollar space via expm1() at call sites.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS

RANDOM_STATE = 42


def to_category_dtype(X):
    """Copy of X with categorical columns cast to pandas 'category' dtype,
    for libraries that expect that (LightGBM, XGBoost, sklearn HGB)."""
    X = X.copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def cat_feature_indices():
    return [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES]


def build_catboost(**overrides):
    from catboost import CatBoostRegressor

    params = dict(
        iterations=1200,
        learning_rate=0.045,
        depth=7,
        l2_leaf_reg=4.0,
        loss_function="MAE",
        random_seed=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=60,
    )
    params.update(overrides)
    return CatBoostRegressor(**params)


def build_lightgbm(**overrides):
    from lightgbm import LGBMRegressor

    params = dict(
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=7,
        num_leaves=63,
        min_child_samples=15,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    params.update(overrides)
    return LGBMRegressor(**params)


def build_xgboost(**overrides):
    from xgboost import XGBRegressor

    params = dict(
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=7,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return XGBRegressor(**params)


def build_hgb(**overrides):
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = dict(
        categorical_features="from_dtype",
        max_iter=600,
        learning_rate=0.04,
        max_depth=4,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return HistGradientBoostingRegressor(**params)


def build_linear_baseline():
    ct = ColumnTransformer(
        [("cat_oh", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline([("ct", ct), ("lr", LinearRegression())])
