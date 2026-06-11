import numpy as np
from numcompute_stream.stats import mean, median, std, quantile, percentile, histogram, variance


# ----------------------------------------------------
# TEST 1: Basic descriptive statistics (mean/median/std/min/max)
# ----------------------------------------------------
def test_basic_descriptive_stats():
    x = np.array([1, 2, 3, 4, 5])

    assert mean(x) == 3.0
    assert median(x) == 3.0
    assert np.isclose(std(x), np.std(x))
    assert np.min(x) == 1
    assert np.max(x) == 5


# ----------------------------------------------------
# TEST 2: Quantiles + NaN handling
# ----------------------------------------------------
def test_quantiles_nan_handling():
    x = np.array([1, 2, np.nan, 4, 5])

    # ignore NaNs
    q1 = quantile(x, 0.25, ignore_nan=True)
    q2 = percentile(x, 50, ignore_nan=True)

    clean = np.array([1, 2, 4, 5])

    assert np.isclose(q1, np.quantile(clean, 0.25))
    assert np.isclose(q2, np.median(clean))


# ----------------------------------------------------
# TEST 3: Histogram + axis-wise behavior + shape checks
# ----------------------------------------------------
def test_histogram_and_axis_behavior():
    x = np.array([1, 2, 2, 3, 4, 4, 4])

    counts, bins = histogram(x, bins=3)

    # histogram shape correctness
    assert len(counts) == 3
    assert len(bins) == 4

    # counts should sum to number of elements
    assert np.isclose(counts.sum(), len(x))

    # axis-wise stats (variance check on 2D input)
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])

    col_var = variance(X, axis=0)
    assert col_var.shape == (3,)
    assert np.all(col_var >= 0)


# ----------------------------------------------------
# BONUS CHECK (optional sanity test)
# ----------------------------------------------------
def test_axis_mean_shape():
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])

    col_mean = mean(X, axis=0)
    row_mean = mean(X, axis=1)

    assert col_mean.shape == (3,)
    assert row_mean.shape == (2,)