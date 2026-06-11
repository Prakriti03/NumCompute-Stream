"""
Unit tests for streaming preprocessing.

Coverage includes:
- StandardScaler partial_fit matching full fit
- MinMaxScaler running min/max updates
- Imputer streaming mean and constant strategies
- OneHotEncoder incremental category expansion
- NaN-heavy chunks and empty input validation
"""

import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Imputer,
    OneHotEncoder,
)


def test_standard_scaler_partial_fit_matches_full_fit():
    X1 = np.array([[1.0, 2.0], [2.0, np.nan]])
    X2 = np.array([[3.0, 4.0], [4.0, 6.0]])
    X_all = np.vstack([X1, X2])

    full = StandardScaler().fit(X_all)
    stream = StandardScaler()
    stream.partial_fit(X1)
    stream.partial_fit(X2)

    assert np.allclose(stream.mean_, full.mean_, equal_nan=False)
    assert np.allclose(stream.std_, full.std_, equal_nan=False)


def test_standard_scaler_handles_all_nan_chunk_without_corrupting_state():
    scaler = StandardScaler()

    scaler.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    old_mean = scaler.mean_.copy()

    scaler.partial_fit(np.array([[np.nan, 6.0], [np.nan, 8.0]]))

    assert np.isclose(scaler.mean_[0], old_mean[0])
    assert np.isclose(scaler.mean_[1], 5.0)


def test_minmax_scaler_partial_fit_updates_global_min_max():
    scaler = MinMaxScaler()

    scaler.partial_fit(np.array([[2.0, 10.0], [4.0, 20.0]]))
    scaler.partial_fit(np.array([[1.0, 30.0], [5.0, 15.0]]))

    assert np.array_equal(scaler.min_, np.array([1.0, 10.0]))
    assert np.array_equal(scaler.max_, np.array([5.0, 30.0]))

    transformed = scaler.transform(np.array([[3.0, 20.0]]))
    assert np.allclose(transformed, np.array([[0.5, 0.5]]))


def test_minmax_scaler_handles_all_nan_chunk_without_corrupting_state():
    scaler = MinMaxScaler()

    scaler.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    old_min = scaler.min_.copy()
    old_max = scaler.max_.copy()

    scaler.partial_fit(np.array([[np.nan, 1.0], [np.nan, 5.0]]))

    assert np.isclose(scaler.min_[0], old_min[0])
    assert np.isclose(scaler.max_[0], old_max[0])
    assert np.isclose(scaler.min_[1], 1.0)
    assert np.isclose(scaler.max_[1], 5.0)


def test_imputer_mean_partial_fit_updates_running_means():
    imputer = Imputer(strategy="mean")

    imputer.partial_fit(np.array([[1.0, np.nan], [3.0, 4.0]]))
    imputer.partial_fit(np.array([[5.0, 6.0], [np.nan, 8.0]]))

    assert np.allclose(imputer.statistics_, np.array([3.0, 6.0]))

    transformed = imputer.transform(np.array([[np.nan, np.nan]]))
    assert np.allclose(transformed, np.array([[3.0, 6.0]]))


def test_imputer_mean_all_nan_chunk_does_not_corrupt_statistics():
    imputer = Imputer(strategy="mean")

    imputer.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    old_stats = imputer.statistics_.copy()

    imputer.partial_fit(np.array([[np.nan, np.nan], [np.nan, np.nan]]))

    assert np.allclose(imputer.statistics_, old_stats)


def test_imputer_constant_strategy():
    imputer = Imputer(strategy="constant", fill_value=-1)

    imputer.partial_fit(np.array([[1.0, np.nan], [np.nan, 2.0]]))
    transformed = imputer.transform(np.array([[np.nan, 5.0]]))

    assert np.array_equal(transformed, np.array([[-1.0, 5.0]]))


def test_onehot_partial_fit_expands_new_categories():
    encoder = OneHotEncoder(handle_unknown="error")

    encoder.partial_fit(np.array([["red"], ["blue"]], dtype=object))
    encoder.partial_fit(np.array([["green"]], dtype=object))

    transformed = encoder.transform(np.array([["red"], ["green"]], dtype=object))

    assert transformed.shape == (2, 3)
    assert np.all(transformed.sum(axis=1) == 1)


def test_onehot_unknown_ignore_returns_zero_vector_for_unknown_column():
    encoder = OneHotEncoder(handle_unknown="ignore")

    encoder.fit(np.array([["red"], ["blue"]], dtype=object))
    transformed = encoder.transform(np.array([["green"]], dtype=object))

    assert transformed.shape == (1, 2)
    assert np.all(transformed == 0)


def test_onehot_unknown_error_raises():
    encoder = OneHotEncoder(handle_unknown="error")

    encoder.fit(np.array([["red"], ["blue"]], dtype=object))

    with pytest.raises(ValueError):
        encoder.transform(np.array([["green"]], dtype=object))


def test_preprocessing_rejects_empty_input():
    with pytest.raises(ValueError):
        StandardScaler().partial_fit(np.empty((0, 2)))

    with pytest.raises(ValueError):
        MinMaxScaler().partial_fit(np.empty((0, 2)))

    with pytest.raises(ValueError):
        Imputer(strategy="mean").partial_fit(np.empty((0, 2)))

    with pytest.raises(ValueError):
        OneHotEncoder().partial_fit(np.empty((0, 2), dtype=object))