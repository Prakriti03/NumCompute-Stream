import numpy as np
import pytest

from numcompute.sort_search import (
    argsort,
    sort,
    topk,
    quickselect,
    binary_search,
    multikey_sort
)



def test_stable_sort():
    X = np.array([3, 1, 2, 1])

    idx1 = argsort(X, stable=True)
    idx2 = np.argsort(X, kind="stable")

    np.testing.assert_array_equal(idx1, idx2)


def test_sort_correctness():
    X = np.array([5, 2, 9, 1])

    sorted_X = sort(X)

    assert np.all(sorted_X == np.sort(X))


def test_multikey_sort():
    X = np.array([
        [2, 10],
        [1, 20],
        [2, 5],
        [1, 15]
    ])

    out = multikey_sort(X, columns=[0, 1])

    # should sort by col0 then col1
    expected = np.array([
        [1, 15],
        [1, 20],
        [2, 5],
        [2, 10]
    ])

    np.testing.assert_array_equal(out, expected)


def test_topk_values_and_indices():
    X = np.array([1, 5, 2, 9, 3])

    values, idx = topk(X, k=2, largest=True)

    # top 2 values should be [9, 5]
    assert set(values.tolist()) == {9, 5}

    # indices must match values
    assert np.all(X[idx] == values)


def test_topk_small_k():
    X = np.array([10, 20, 30, 40])

    values, _ = topk(X, k=1, largest=False)

    assert values[0] == 10


def test_quickselect_median():
    X = np.array([7, 1, 3, 5, 9])

    median = quickselect(X, k=2)

    assert median == 5


def test_quickselect_edges():
    X = np.array([10, 20, 30])

    assert quickselect(X, 0) == 10
    assert quickselect(X, 2) == 30



def test_topk_invalid_k():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        topk(X, k=0)

    with pytest.raises(ValueError):
        topk(X, k=10)