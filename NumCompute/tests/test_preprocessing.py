import numpy as np
import pytest

from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    Imputer
)

def test_standard_scaler_zscore():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # mean should be ~0 (centered)
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)

    # std should be ~1 (scaled)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)


def test_minmax_scaler_range():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0]
    ])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # check range [0,1]
    assert np.all(X_scaled >= 0)
    assert np.all(X_scaled <= 1)

    # min should map to 0
    assert np.allclose(X_scaled.min(axis=0), 0)

    # max should map to 1
    assert np.allclose(X_scaled.max(axis=0), 1)


def test_one_hot_encoder_correctness():
    X = np.array([
        ["cat", "A"],
        ["dog", "B"],
        ["cat", "B"]
    ])

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    # expected shape = (n_samples, total_categories)
    assert X_encoded.shape[0] == 3

    # each row should have exactly 2 ones (2 features → one-hot per feature)
    assert np.all(X_encoded.sum(axis=1) == 2)

    # binary output check
    assert set(np.unique(X_encoded)).issubset({0.0, 1.0})


def test_imputer_mean_strategy():
    X = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
        [np.nan, 6.0]
    ])

    imputer = Imputer(strategy="mean")
    X_imp = imputer.fit_transform(X)

    # no NaNs after transform
    assert not np.isnan(X_imp).any()

    # check column-wise mean imputation
    assert X_imp[0, 1] == pytest.approx(5.0)  # mean of [4,6]