"""
tests/test_benchmarking.py — Basic tests for benchmarking.py
"""

import pytest
import numpy as np

from numcompute.benchmarking import (
    benchmark,
    scaling_benchmark,
    _loop_mean,
    _loop_std,
    _loop_mse,
    _loop_topk,
    run_all_benchmarks,
)

def test_benchmark_returns_expected_keys():
    result = benchmark(np.sum, args=(np.ones(100),), n_runs=3, warmup=1)

    assert isinstance(result, dict)
    assert "name" in result
    assert "mean" in result
    assert "std" in result
    assert "min" in result
    assert result["mean"] > 0

def test_benchmark_custom_name():
    result = benchmark(np.sum, args=(np.ones(10),), n_runs=3, name="my_sum")

    assert result["name"] == "my_sum"

def test_benchmark_rejects_invalid_input():
    with pytest.raises(TypeError):
        benchmark("not a function")

    with pytest.raises(ValueError):
        benchmark(np.sum, args=(np.ones(10),), n_runs=0)

def test_scaling_benchmark_returns_table():
    def loop_mean(x):
        return sum(x) / len(x)

    gen = lambda n: (np.ones(n),)

    out = scaling_benchmark(
        np.mean,
        loop_mean,
        sizes=[100, 500],
        data_gen=gen,
        n_runs=3,
        warmup=1,
    )

    assert isinstance(out, dict)
    assert len(out["speedups"]) == 2
    assert isinstance(out["table"], str)
    assert "Size" in out["table"]
    assert "Speedup" in out["table"]

def test_scaling_benchmark_empty_sizes_raises():
    with pytest.raises(ValueError):
        scaling_benchmark(np.mean, np.mean, [], lambda n: (np.ones(n),))

def test_loop_baselines_match_numpy():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    assert abs(_loop_mean(x) - np.mean(x)) < 1e-10
    assert abs(_loop_std(x) - np.std(x, ddof=0)) < 1e-10
    assert abs(_loop_mse(x, y_pred) - np.mean((x - y_pred) ** 2)) < 1e-10

def test_loop_topk_returns_largest_values():
    x = np.array([3.0, 1.0, 4.0, 5.0, 2.0])

    result = _loop_topk(x, k=3)

    np.testing.assert_array_equal(result, [5.0, 4.0, 3.0])

def test_run_all_benchmarks_returns_results(capsys):
    results = run_all_benchmarks(sizes=[100], n_runs=3)

    captured = capsys.readouterr().out

    assert isinstance(results, dict)
    assert "mean" in results
    assert "std" in results
    assert "top-10" in results
    assert "mse" in results
    assert "Python" in captured
    assert "NumPy" in captured
    assert "Speedup" in captured