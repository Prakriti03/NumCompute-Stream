import numpy as np
from numcompute.rank import rank, percentile


# =========================
# 1. Rank: tie handling (average, dense, ordinal)
# =========================
def test_rank_tie_handling_all_methods():
    data = np.array([3, 1, 4, 1, 5])

    avg = rank(data, method="average")
    dense = rank(data, method="dense")
    ordinal = rank(data, method="ordinal")

    # --- average rank: ties (1,1) should have same rank 1.5 ---
    assert np.isclose(avg[1], 1.5)
    assert np.isclose(avg[3], 1.5)

    # --- dense rank: ties share same rank, no gaps ---
    assert dense[1] == dense[3]

    # --- ordinal rank: strictly increasing 1..n ---
    assert np.array_equal(np.sort(ordinal), np.array([1, 2, 3, 4, 5], dtype=float))


# =========================
# 2. Percentile: interpolation correctness
# =========================
def test_percentile_interpolation_modes():
    data = np.array([1, 2, 3, 4, 5])

    # 50th percentile (median)
    assert np.isclose(percentile(data, 50, interpolation="linear"), 3.0)

    # lower interpolation → floor index
    assert percentile(data, 50, interpolation="lower") == 3.0

    # higher interpolation → ceil index
    assert percentile(data, 50, interpolation="higher") == 3.0

    # midpoint interpolation → average of neighbors
    mid = percentile(data, 50, interpolation="midpoint")
    assert np.isclose(mid, 3.0)


# =========================
# 3. Edge cases: multidimensional + NaNs + multiple percentiles
# =========================
def test_rank_and_percentile_edge_cases():
    # --- rank on 2D input (flattening behavior) ---
    data_2d = np.array([[3, 1], [4, 1]])

    r = rank(data_2d, method="dense")
    assert r.shape == data_2d.shape

    # --- percentile with multiple q values ---
    data = np.array([1, 2, 3, 4, 5])
    q_vals = np.array([0, 50, 100])

    p = percentile(data, q_vals, interpolation="linear")
    assert p.shape == (3,)
    assert np.isclose(p[0], 1.0)
    assert np.isclose(p[-1], 5.0)

    # --- NaN handling ---
    data_nan = np.array([1, np.nan, 3, 4])
    p_nan = percentile(data_nan, 50, ignore_nan=True)
    assert not np.isnan(p_nan)