import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import (
    ALL_FEATURES, TARGET_COL,
    build_preprocessor, fit_preprocessor, apply_preprocessor,
    get_X_y, split_data,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 50
    data = {"Time": np.arange(n, dtype=float), "Amount": np.random.uniform(1, 500, n)}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)
    data[TARGET_COL] = [1, 1] + [0] * (n - 2)
    return pd.DataFrame(data)


def test_split_preserves_total_rows(sample_df):
    train, val, test = split_data(sample_df)
    assert len(train) + len(val) + len(test) == len(sample_df)


def test_split_no_overlap(sample_df):
    train, val, test = split_data(sample_df)
    assert set(train.index).isdisjoint(set(val.index))
    assert set(train.index).isdisjoint(set(test.index))
    assert set(val.index).isdisjoint(set(test.index))


def test_preprocessor_output_shape(sample_df):
    train, val, _ = split_data(sample_df)
    prep = fit_preprocessor(train)
    X = apply_preprocessor(prep, val)
    assert X.shape[1] == len(ALL_FEATURES)


def test_preprocessor_returns_float(sample_df):
    train, val, _ = split_data(sample_df)
    prep = fit_preprocessor(train)
    X = apply_preprocessor(prep, val)
    assert X.dtype in (np.float32, np.float64)


def test_get_X_y(sample_df):
    X, y = get_X_y(sample_df)
    assert list(X.columns) == ALL_FEATURES
    assert y.name == TARGET_COL
    assert len(X) == len(y)
