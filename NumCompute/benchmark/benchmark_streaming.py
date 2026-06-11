"""
Benchmark script for NumCompute-Stream.

Covers:
1. Loop vs vectorised NumPy performance.
2. Streaming DecisionTreeClassifier vs RandomForestClassifier.
3. Saves benchmark results and plots for report evidence.
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler, Imputer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.metrics import StreamingAccuracy


OUT_DIR = ROOT / "benchmark" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_dataset(n_samples=2000, n_features=6, random_state=42):
    rng = np.random.default_rng(random_state)

    X = rng.normal(size=(n_samples, n_features))
    noise = rng.normal(scale=1.5, size=n_samples)

    score = (
        1.2 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.5 * X[:, 2] ** 2           # nonlinear term
        + 0.3 * X[:, 3] * X[:, 4]      # interaction term
        + noise
    )

    y = (score > np.median(score)).astype(int)

    missing_mask = rng.random(X.shape) < 0.03
    X[missing_mask] = np.nan

    return X, y


def time_function(fn, repeats=5):
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return float(np.median(times)), result


def loop_accuracy(y_true, y_pred):
    correct = 0

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    return correct / len(y_true)


def vectorized_accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def loop_column_mean(X):
    means = []

    for j in range(X.shape[1]):
        total = 0.0
        count = 0

        for i in range(X.shape[0]):
            value = X[i, j]
            if not np.isnan(value):
                total += value
                count += 1

        means.append(total / count)

    return np.asarray(means)


def vectorized_column_mean(X):
    return np.nanmean(X, axis=0)


def make_tree_pipeline():
    return Pipeline([
        ("impute", Imputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(
            max_depth=8,
            criterion="gini",
            random_state=42,
        )),
    ])


def make_forest_pipeline():
    return Pipeline([
        ("impute", Imputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=10,
            max_depth=8,
            criterion="gini",
            random_state=42,
        )),
    ])


def benchmark_loop_vs_vectorized(X, y):
    rng = np.random.default_rng(42)
    y_pred = rng.integers(0, 2, size=y.shape[0])

    rows = []

    loop_time, loop_result = time_function(lambda: loop_accuracy(y, y_pred))
    vec_time, vec_result = time_function(lambda: vectorized_accuracy(y, y_pred))

    rows.append({
        "benchmark": "accuracy",
        "implementation": "python_loop",
        "time_sec": loop_time,
        "result": loop_result,
    })

    rows.append({
        "benchmark": "accuracy",
        "implementation": "numpy_vectorized",
        "time_sec": vec_time,
        "result": vec_result,
    })

    loop_time, loop_result = time_function(lambda: loop_column_mean(X))
    vec_time, vec_result = time_function(lambda: vectorized_column_mean(X))

    rows.append({
        "benchmark": "column_mean",
        "implementation": "python_loop",
        "time_sec": loop_time,
        "result": float(np.mean(loop_result)),
    })

    rows.append({
        "benchmark": "column_mean",
        "implementation": "numpy_vectorized",
        "time_sec": vec_time,
        "result": float(np.mean(vec_result)),
    })

    return rows


def benchmark_streaming_models(X, y, chunk_size=100):
    # Hold out last 20% for honest evaluation
    n_train = int(len(X) * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    tree_trainer   = StreamTrainer(make_tree_pipeline(),   metrics={"accuracy": StreamingAccuracy()})
    forest_trainer = StreamTrainer(make_forest_pipeline(), metrics={"accuracy": StreamingAccuracy()})

    tree_logs   = []
    forest_logs = []

    for start in range(0, X_train.shape[0], chunk_size):
        Xc, yc = X_train[start:start+chunk_size], y_train[start:start+chunk_size]
        tree_log   = tree_trainer.fit_chunk(Xc, yc, classes=np.array([0, 1]))
        forest_log = forest_trainer.fit_chunk(Xc, yc, classes=np.array([0, 1]))

        # Replace chunk_accuracy with holdout accuracy
        tree_log["chunk_accuracy"]   = float(np.mean(tree_trainer.pipeline.predict(X_test) == y_test))
        forest_log["chunk_accuracy"] = float(np.mean(forest_trainer.pipeline.predict(X_test) == y_test))

        tree_logs.append(tree_log)
        forest_logs.append(forest_log)

    rows = []
    for log in tree_logs:
        rows.append({"benchmark":"streaming_model","implementation":"decision_tree",
                     "chunk":log["chunk"],"chunk_size":log["chunk_size"],
                     "time_sec":log["train_time_sec"],"holdout_accuracy":log["chunk_accuracy"],
                     "memory_bytes":log["memory_bytes"]})
    for log in forest_logs:
        rows.append({"benchmark":"streaming_model","implementation":"random_forest",
                     "chunk":log["chunk"],"chunk_size":log["chunk_size"],
                     "time_sec":log["train_time_sec"],"holdout_accuracy":log["chunk_accuracy"],
                     "memory_bytes":log["memory_bytes"]})

    return rows, tree_logs, forest_logs


def save_csv(rows, path):
    keys = sorted(set().union(*(row.keys() for row in rows)))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_loop_vs_vectorized(rows):
    # Group by benchmark name
    benchmarks = {}
    for row in rows:
        benchmarks.setdefault(row["benchmark"], {})[row["implementation"]] = row["time_sec"]

    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    colors = {"python_loop": "#e05555", "numpy_vectorized": "#4a90d9"}

    for ax, (bench_name, times) in zip(axes, benchmarks.items()):
        impls = list(times.keys())
        vals  = [times[k] for k in impls]
        bars  = ax.bar(impls, vals, color=[colors.get(k, "steelblue") for k in impls])

        # Add speedup annotation
        if "python_loop" in times and "numpy_vectorized" in times:
            speedup = times["python_loop"] / times["numpy_vectorized"]
            ax.set_title(f"{bench_name}\n({speedup:.0f}x speedup)", fontweight="bold")
        else:
            ax.set_title(bench_name)

        ax.set_ylabel("Median time (seconds)")
        ax.set_xlabel("Implementation")

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.5f}s", ha="center", va="bottom", fontsize=9)

        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Loop vs NumPy vectorised benchmark", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loop_vs_vectorized.png", bbox_inches="tight")
    plt.close(fig)

def plot_streaming_accuracy(tree_logs, forest_logs):
    tree_acc   = np.array([log["chunk_accuracy"] for log in tree_logs])    # ← was cumulative_accuracy
    forest_acc = np.array([log["chunk_accuracy"] for log in forest_logs])  # ← was cumulative_accuracy
    chunks = np.arange(1, len(tree_acc) + 1)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(chunks, tree_acc, marker="o", label="Decision tree")
    plt.plot(chunks, forest_acc, marker="s", label="Random forest")
    plt.xlabel("Chunk")
    plt.ylabel("Holdout accuracy (test set = 20% held out)")
    plt.title("Streaming benchmark: tree vs forest holdout accuracy (20% held out)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = OUT_DIR / "streaming_accuracy.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_streaming_train_time(tree_logs, forest_logs):
    tree_time = np.array([log["train_time_sec"] for log in tree_logs])
    forest_time = np.array([log["train_time_sec"] for log in forest_logs])
    chunks = np.arange(1, len(tree_time) + 1)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(chunks, tree_time, marker="o", label="Decision tree")
    plt.plot(chunks, forest_time, marker="s", label="Random forest")
    plt.xlabel("Chunk")
    plt.ylabel("Training time (seconds)")
    plt.title("Streaming model benchmark: training time per chunk")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = OUT_DIR / "streaming_train_time.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_training_time_growth(tree_logs, forest_logs):
    tree_time   = np.array([log["train_time_sec"] for log in tree_logs])
    forest_time = np.array([log["train_time_sec"] for log in forest_logs])
    samples_seen = np.array([log["chunk"] * log["chunk_size"] for log in tree_logs])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: linear scale
    axes[0].plot(samples_seen, tree_time,   marker="o", label="Decision tree")
    axes[0].plot(samples_seen, forest_time, marker="s", label="Random forest")
    axes[0].set_xlabel("Samples seen (accumulated)")
    axes[0].set_ylabel("Training time (seconds)")
    axes[0].set_title("Training time growth (linear)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Right: log-log scale — shows O(n^k) growth rate clearly
    axes[1].plot(samples_seen, tree_time,   marker="o", label="Decision tree")
    axes[1].plot(samples_seen, forest_time, marker="s", label="Random forest")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Samples seen (log scale)")
    axes[1].set_ylabel("Training time (log scale)")
    axes[1].set_title("Training time growth (log-log: shows O(n^k) slope)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training time growth: accumulate-and-rebuild streaming strategy",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_time_growth.png", bbox_inches="tight")
    plt.close(fig)

def main():
    X, y = make_dataset(n_samples=2000, n_features=6, random_state=42)

    loop_rows = benchmark_loop_vs_vectorized(X, y)
    streaming_rows, tree_logs, forest_logs = benchmark_streaming_models(
        X,
        y,
        chunk_size=100,
    )

    save_csv(loop_rows, OUT_DIR / "loop_vs_vectorized_results.csv")
    save_csv(streaming_rows, OUT_DIR / "streaming_model_results.csv")

    plot_loop_vs_vectorized(loop_rows)
    plot_streaming_accuracy(tree_logs, forest_logs)
    plot_streaming_train_time(tree_logs, forest_logs)
    plot_training_time_growth(tree_logs, forest_logs)

    print("Benchmark complete.")
    print(f"Results saved in: {OUT_DIR}")
    print()
    
    # ── Loop vs vectorised ──────────────────────────────────────────────
    benchmarks = {}
    for row in loop_rows:
        benchmarks.setdefault(row["benchmark"], {})[row["implementation"]] = row["time_sec"]

    print("Loop vs NumPy vectorised speedup:")
    print(f"{'Benchmark':<20} {'Loop (s)':>12} {'NumPy (s)':>12} {'Speedup':>10}")
    print("-" * 58)
    for name, times in benchmarks.items():
        loop_t = times["python_loop"]
        vec_t  = times["numpy_vectorized"]
        speedup = loop_t / vec_t if vec_t > 0 else float("inf")
        print(f"{name:<20} {loop_t:>12.6f} {vec_t:>12.6f} {speedup:>9.1f}x")

    print()
    
    # ── Streaming models ────────────────────────────────────────────────
    print("Final cumulative accuracy:")
    print(f"Decision tree:  {tree_logs[-1]['cumulative_accuracy']:.4f}")
    print(f"Random forest:  {forest_logs[-1]['cumulative_accuracy']:.4f}")
    print()
    print("Median streaming train time:")
    print(f"Decision tree:  {np.median([x['train_time_sec'] for x in tree_logs]):.6f} sec/chunk")
    print(f"Random forest:  {np.median([x['train_time_sec'] for x in forest_logs]):.6f} sec/chunk")
    print(f"  Forest overhead: {np.median([x['train_time_sec'] for x in forest_logs]) / np.median([x['train_time_sec'] for x in tree_logs]):.1f}x slower than single tree")


if __name__ == "__main__":
    main()