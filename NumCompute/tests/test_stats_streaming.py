"""
Unit tests for streaming statistics.

Coverage includes:
- Welford mean/variance against NumPy
- NaN-aware per-feature updates
- StreamingHistogram accumulation and sliding window
- StreamingQuantile updates
- StreamingStats summary/reset behaviour
- invalid streaming parameters
"""

import numpy as np
import pytest

from numcompute_stream.stats import (
    WelfordStatistics,
    StreamingHistogram,
    StreamingQuantile,
    StreamingStats,
)


def test_welford_update_stats_matches_numpy_mean_variance():
    X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    X_all = np.vstack([X1, X2])

    stats = WelfordStatistics()
    stats.update_stats(X1)
    stats.update_stats(X2)

    assert np.allclose(stats.mean(), np.mean(X_all, axis=0))
    assert np.allclose(stats.variance(), np.var(X_all, axis=0))


def test_welford_ignores_nan_per_feature():
    X1 = np.array([[1.0, 2.0], [3.0, np.nan]])
    X2 = np.array([[np.nan, 6.0], [5.0, 8.0]])

    stats = WelfordStatistics()
    stats.update_stats(X1)
    stats.update_stats(X2)

    X_all = np.vstack([X1, X2])

    assert np.allclose(stats.mean(), np.nanmean(X_all, axis=0))
    assert np.allclose(stats.variance(), np.nanvar(X_all, axis=0))


def test_streaming_histogram_accumulates_counts():
    hist = StreamingHistogram(bins=2)

    hist.update_stats(np.array([0.0, 1.0]))
    hist.update_stats(np.array([2.0, 3.0]))

    counts, edges = hist.result()

    assert counts.sum() == 4
    assert len(edges) == 3


def test_streaming_histogram_window_size_keeps_recent_values():
    hist = StreamingHistogram(bins=2, window_size=3)

    hist.update_stats(np.array([1.0, 2.0, 3.0]))
    hist.update_stats(np.array([4.0, 5.0]))

    counts, _ = hist.result()

    assert counts.sum() == 3


def test_streaming_quantile_matches_numpy_quantile():
    q = StreamingQuantile([0.25, 0.5, 0.75])

    q.update_stats(np.array([1.0, 2.0]))
    q.update_stats(np.array([3.0, 4.0]))

    assert np.allclose(q.result(), np.quantile(np.array([1.0, 2.0, 3.0, 4.0]), [0.25, 0.5, 0.75]))


def test_streaming_quantile_ignores_nan():
    q = StreamingQuantile(0.5)

    q.update_stats(np.array([1.0, np.nan, 3.0]))

    assert np.isclose(q.result(), 2.0)


def test_streaming_stats_summary_after_chunks():
    stats = StreamingStats(n_features=2, bins=3)

    stats.update_stats(np.array([[1.0, 2.0], [3.0, 4.0]]))
    stats.update_stats(np.array([[5.0, 6.0]]))

    summary = stats.summary()

    assert summary["n_chunks"] == 2
    assert np.allclose(summary["mean"], np.array([3.0, 4.0]))
    assert summary["quantiles"].shape[0] == 2


def test_streaming_stats_rejects_wrong_feature_count():
    stats = StreamingStats(n_features=2)

    with pytest.raises(ValueError):
        stats.update_stats(np.array([[1.0, 2.0, 3.0]]))


def test_streaming_classes_reset():
    stats = StreamingStats(n_features=1)

    stats.update_stats(np.array([[1.0], [2.0]]))
    stats.reset()

    assert stats.n_chunks_seen == 0
    assert stats.mean() is None


def test_invalid_streaming_parameters_raise_error():
    with pytest.raises(ValueError):
        StreamingHistogram(bins=0)

    with pytest.raises(ValueError):
        StreamingQuantile(1.5)

    with pytest.raises(ValueError):
        StreamingStats(n_features=0)