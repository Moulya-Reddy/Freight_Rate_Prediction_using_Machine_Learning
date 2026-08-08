"""
Forward-chaining (expanding-window) time series cross-validation.

A single train/holdout split (train on Jan-Aug, validate Sep-Oct) is what
the original baseline used, and it's kept as the headline number for the
report because it mirrors the actual train_test.csv -> validation.csv gap
most directly. But tuning hyperparameters against a single holdout window
risks overfitting the hyperparameters to that particular 8-week slice.

For hyperparameter search we instead use 3 forward-chaining folds, each
validating on a different, later slice of the year, with the training set
growing each time (never shrinking, never including future data relative
to its own validation slice). The final fold's window matches the
headline holdout so the two numbers stay comparable.
"""
from __future__ import annotations

import pandas as pd

FOLD_BOUNDARIES = [
    # (train_end_exclusive, valid_end_exclusive)
    ("2025-08-04", "2025-09-01"),
    ("2025-09-01", "2025-09-29"),
    ("2025-09-29", "2025-11-01"),  # matches the headline 8-week holdout
]


def make_folds(df: pd.DataFrame, boundaries=None):
    """Yield (train_df, valid_df) pairs. `df` must be sorted by date and
    already have causal lane-history features computed over its full span."""
    boundaries = boundaries if boundaries is not None else FOLD_BOUNDARIES
    for train_end, valid_end in boundaries:
        train_end_ts = pd.Timestamp(train_end)
        valid_end_ts = pd.Timestamp(valid_end)
        train_fold = df[df["date"] < train_end_ts]
        valid_fold = df[(df["date"] >= train_end_ts) & (df["date"] < valid_end_ts)]
        yield train_fold, valid_fold
