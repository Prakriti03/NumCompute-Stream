"""
benchmarking.py - Timing comparisons for NumCompute vectorised vs loop implementations.

Provides a timing helper, Python loop baselines, and a full benchmark
suite comparing numcompute functions against loop equivalents across
increasing input sizes. Includes environment info for reproducibility.

"""
from __future__ import annotations
import time
import platform
import sys
import numpy as np
from numpy.typing import NDArray

from numcompute.stats import mean, std
from numcompute.metrics import mse
from numcompute.sort_search import topk


def benchmark(func, args=(), n_runs=20, warmup=3, name=None):
    """Time one function over repeated runs.
    Parameters
    ----------
    func : callable
        Function to benchmark.
    args : tuple
        Positional arguments passed to func.
    n_runs : int
        Number of timed runs.
    warmup : int
        Number of untimed warm-up runs before timing.
    name : str or None
        Optional name used in the returned result.
    Returns
    -------
    dict
        Dictionary with name, mean runtime, standard deviation, and minimum time.
    Raises
    ------
    TypeError
        If func is not callable.
    ValueError
        If n_runs is less than 1.
    Notes
    -----
    Time complexity depends on the function being benchmarked.
    Space complexity is O(n_runs) for storing timings.
    """
    if not callable(func):
        raise TypeError("func must be callable.")
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}.")

    name = name or getattr(func, "__name__", "unknown")

    for _ in range(warmup):
        func(*args)

    times = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        t0 = time.perf_counter()
        func(*args)
        times[i] = time.perf_counter() - t0

    return {
        "name": name,
        "mean": float(np.mean(times)),
        "std": float(np.std(times, ddof=1)) if n_runs > 1 else 0.0,
        "min": float(np.min(times)),
    }

def scaling_benchmark(func_a, func_b, sizes, data_gen, n_runs=10, warmup=2, name_a="vectorised", name_b="loop"):
    """Compare two functions across increasing input sizes.
    Parameters
    ----------
    func_a : callable
        First function, usually the vectorised NumCompute implementation.
    func_b : callable
        Second function, usually the Python loop baseline.
    sizes : list of int
        Input sizes to benchmark.
    data_gen : callable
        Function that takes n and returns arguments for the benchmarked functions.
    n_runs : int
        Number of timed runs for each size.
    warmup : int
        Number of untimed warm-up runs.
    name_a : str
        Display name for func_a.
    name_b : str
        Display name for func_b.
    Returns
    -------
    dict
        Dictionary containing sizes, mean timings, speedups, and a formatted table.
    Raises
    ------
    ValueError - If sizes is empty.
    Notes
    -----
    Time complexity depends on the benchmarked functions and number of sizes.
    Space complexity is O(len(sizes)) for storing timing results.
    """
    if not sizes:
        raise ValueError("sizes must not be empty.")

    times_a, times_b, speedups = [], [], []
    for n in sizes:
        args = data_gen(n)
        if not isinstance(args, tuple):
            args = (args,)

        ra = benchmark(func_a, args, n_runs=n_runs, warmup=warmup, name=name_a)
        rb = benchmark(func_b, args, n_runs=n_runs, warmup=warmup, name=name_b)

        times_a.append(ra["mean"])
        times_b.append(rb["mean"])
        speedups.append(rb["mean"] / ra["mean"] if ra["mean"] > 0 else float("inf"))

    header = f"{'Size':<12} {name_a + ' (s)':<18} {name_b + ' (s)':<18} {'Speedup':<10}"
    sep = "-" * len(header)
    rows = [header, sep]
    for n, ta, tb, sp in zip(sizes, times_a, times_b, speedups):
        rows.append(f"{n:<12} {ta:<18.6f} {tb:<18.6f} {sp:<10.2f}x")

    return {
        "sizes": sizes,
        "times_a": times_a,
        "times_b": times_b,
        "speedups": speedups,
        "table": "\n".join(rows),
    }

def _loop_mean(x):
    """Compute mean using a Python loop baseline.
    Parameters
    ----------
    x : array, shape (n,)
        Input values.
    Returns
    -------
    float
        Mean of x.
    Notes
    -----
    Time complexity: O(n).
    Space complexity: O(1).
    """
    total = 0.0
    for v in x:
        total += float(v)
    return total / len(x)

def _loop_std(x):
    """Compute population standard deviation using Python loops.
    Parameters
    ----------
    x : array, shape (n,)
        Input values.
    Returns
    -------
    float
        Population standard deviation of x.
    Notes
    -----
    Time complexity: O(n).
    Space complexity: O(1).
    """
    m = _loop_mean(x)
    total = 0.0
    for v in x:
        total += (float(v) - m) ** 2
    return (total / len(x)) ** 0.5

def _loop_topk(x, k=10):
    """Return k largest values using a Python sorting baseline.
    Parameters
    ----------
    x : array, shape (n,)
        Input values.
    k : int
        Number of largest values to return.
    Returns
    -------
    ndarray, shape (k,)
        Largest k values in descending order.
    Notes
    -----
    Time complexity: O(n log n), due to full sorting.
    Space complexity: O(n).
    """
    return np.asarray(sorted((float(v) for v in x), reverse=True)[:k], dtype=np.float64)

def _loop_mse(y_true, y_pred):
    """Compute mean squared error using Python loops.
    Parameters
    ----------
    y_true : array, shape (n,)
        True target values.
    y_pred : array, shape (n,)
        Predicted target values.
    Returns
    -------
    float
        Mean squared error.
    Notes
    -----
    Time complexity: O(n).
    Space complexity: O(1).
    """
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        total += (float(yt) - float(yp)) ** 2
    return total / len(y_true)


def print_environment():
    """Print Python, NumPy, and OS info for reproducibility."""
    print("Environment")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  NumPy  : {np.__version__}")
    print(f"  OS     : {platform.system()} {platform.release()}")


def run_all_benchmarks(sizes=None, n_runs=20, seed=42):
    """Run the full benchmark suite.
    Compares selected NumCompute vectorised implementations against
    Python loop baselines for mean, standard deviation, top-k, and MSE.
    Parameters
    ----------
    sizes : list of int or None
        Input sizes to benchmark. If None, defaults to [1000, 10000, 100000].
    n_runs : int
        Number of timed runs for each benchmark.
    seed : int
        Random seed used to generate reproducible input data.
    Returns
    -------
    dict
        Benchmark results keyed by benchmark name.
    Notes
    -----
    Prints environment information and timing tables for reproducibility.
    Time complexity depends on the functions being benchmarked.
    Space complexity depends on the generated input size.
    """
    rng = np.random.default_rng(seed)
    sizes = sizes or [1_000, 10_000, 100_000]
    k = 10

    suite = [
        ("mean",mean,_loop_mean,  lambda n: (rng.standard_normal(n),)),
        ("std",std,_loop_std,lambda n: (rng.standard_normal(n),)),
        (f"top-{k}", lambda x: topk(x, k, largest=True)[0], lambda x: _loop_topk(x, k), lambda n: (rng.standard_normal(n),)),
        ("mse",mse,_loop_mse,lambda n: (rng.standard_normal(n), rng.standard_normal(n))),
    ]

    print_environment()
    print()
    print("-" * 66)
    print(" NumCompute — vectorised vs Python loop benchmarks")
    print("-" * 66)

    results = {}
    for label, vec_fn, loop_fn, gen in suite:
        print(f"\n── {label} ──")
        out = scaling_benchmark(
            vec_fn, loop_fn,
            sizes=sizes, data_gen=gen,
            n_runs=n_runs, warmup=3,
            name_a="numcompute", name_b="loop",
        )
        print(out["table"])
        results[label] = out

    print("\n" + "=" * 66)
    print("Summary — mean speedup across all sizes:")
    for label, out in results.items():
        avg = float(np.mean(out["speedups"]))
        print(f"  {label:<12} {avg:.2f}x faster with numcompute")
    print("=" * 66)

    return results


if __name__ == "__main__":
    run_all_benchmarks()