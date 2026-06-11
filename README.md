# NumCompute-Stream

A tree-based, streaming-compatible machine learning framework built from scratch using NumPy only.  
**Assignment 2.2 — Prakriti Timalsena | Adelaide University | 2026**

---

## Requirements

- Python >= 3.9
- NumPy >= 1.21
- matplotlib (for visualisation and demo)
- jupyter (for the demo notebook)
- pytest (for tests)

---

## Setup

All commands below assume you are inside the `NumCompute` folder.

```bash
cd NumCompute
```

### Install the package (editable mode)

```bash
pip install -e .
```

This installs `numcompute_stream` as an importable package from the current directory. You only need to do this once.

Verify the install worked:

```bash
python -c "import numcompute_stream; print('OK')"
```

---

## Running the Tests

```bash
pytest tests/ -v
```

Expected output: **149 passed**

To run a specific test file:

```bash
pytest tests/test_tree.py -v          # 52 tests — DecisionTreeClassifier
pytest tests/test_ensemble.py -v      # 12 tests — RandomForestClassifier
pytest tests/test_stream.py -v        # 12 tests — StreamTrainer
pytest tests/test_visualise.py -v     # 13 tests — visualise.py
pytest tests/test_preprocessing_streaming.py -v  # 11 tests — partial_fit scalers
pytest tests/test_stats_streaming.py -v          # 10 tests — StreamingStats / WelfordStatistics
pytest tests/test_pipeline_streaming.py -v       #  8 tests — Pipeline.partial_fit
pytest tests/test_metrics_streaming.py -v        #  5 tests — StreamingAUC / RollingAccuracy
```

To run quietly (summary only):

```bash
pytest tests/ -q
```

---

## Running the Demo

The demo notebook simulates a real streaming scenario: loads a CSV dataset, splits it into chunks, trains two models incrementally, and visualises metrics.

```bash
cd demo
jupyter notebook stream_demo.ipynb
```

Or open it in VS Code / JupyterLab. Run all cells top-to-bottom.

The notebook will:
1. Generate and save `stream_demo_dataset.csv`
2. Load it using the custom `CSVReader`
3. Train a `DecisionTreeClassifier` pipeline and a `RandomForestClassifier` pipeline chunk-by-chunk using `partial_fit`
4. Log per-chunk accuracy, training time, and memory footprint
5. Compare Gini vs Entropy criterion
6. Plot all metrics using `visualise.py`
7. Verify streaming statistics match NumPy batch results exactly

---

## Running the Benchmark

```bash
python benchmark/benchmark_streaming.py
```

This will:
- Benchmark Python loop vs NumPy vectorised implementations (accuracy, column mean)
- Benchmark streaming `DecisionTreeClassifier` vs `RandomForestClassifier` on a held-out test set over 16 chunks
- Save four plots and two CSVs to `benchmark/results/`

Results are saved to:

```
benchmark/results/
├── loop_vs_vectorized.png
├── loop_vs_vectorized_results.csv
├── streaming_accuracy.png
├── streaming_train_time.png
├── streaming_model_results.csv
└── training_time_growth.png
```
---

## Folder Structure

```
NumCompute/
│
├── numcompute_stream/          # Main package
│   ├── __init__.py             # Public API exports
│   ├── tree.py                 # DecisionTreeClassifier (Gini/entropy, partial_fit)
│   ├── ensemble.py             # RandomForestClassifier / EnsembleClassifier
│   ├── preprocessing.py        # StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
│   ├── metrics.py              # Batch + streaming metrics (Accuracy, Precision, Recall,
│   │                           #   F1, AUC, ConfusionMatrix, RollingAccuracy)
│   ├── stats.py                # WelfordStatistics, StreamingHistogram,
│   │                           #   StreamingQuantile, StreamingStats, describe()
│   ├── stream.py               # StreamTrainer (fit_chunk, score_chunk, logging)
│   ├── pipeline.py             # Pipeline (partial_fit, predict, score)
│   ├── visualise.py            # plot_metric_over_time, compare_models,
│   │                           #   plot_predictions_vs_ground_truth
│   ├── io.py                   # CSVReader with chunked streaming
│   └── utils.py                # euclidean_distance, cosine_similarity, softmax, relu
│
├── tests/                      # Unit tests (149 total, all passing)
│   ├── test_tree.py            # 52 tests
│   ├── test_ensemble.py        # 12 tests
│   ├── test_stream.py          # 12 tests
│   ├── test_visualise.py       # 13 tests
│   ├── test_preprocessing_streaming.py  # 11 tests
│   ├── test_stats_streaming.py          # 10 tests
│   ├── test_pipeline_streaming.py       #  8 tests
│   ├── test_metrics_streaming.py        #  5 tests
│   └── legacy/                 # Tests carried over from Assignment 2.1
│       ├── test_io.py
│       ├── test_metrics.py
│       ├── test_pipeline.py
│       ├── test_stats.py
│       └── test_utils.py
│
├── demo/                       # Demo notebook and datasets
│   ├── stream_demo.ipynb       # Main streaming demo (required submission)
│   ├── stream_demo_dataset.csv # Auto-generated by the notebook
│   ├── quickstart.ipynb        # Short quickstart examples
│   ├── titanic_dataset.csv
│   └── Mall_Customers.csv
│
├── benchmark/                  # Benchmarking
│   ├── benchmark_streaming.py  # Loop vs vectorised + streaming model comparison
│   └── results/                # Auto-generated plots and CSVs
│       ├── loop_vs_vectorized.png
│       ├── loop_vs_vectorized_results.csv
│       ├── streaming_accuracy.png
│       ├── streaming_train_time.png
│       ├── streaming_model_results.csv
│       └── training_time_growth.png
│
└── pyproject.toml              # Package metadata and dependencies
```

---

## Module Summary

| Module | Key classes / functions | Streaming |
|---|---|---|
| `tree.py` | `DecisionTreeClassifier` | `partial_fit` ✅ |
| `ensemble.py` | `RandomForestClassifier`, `EnsembleClassifier` | `partial_fit` ✅ |
| `preprocessing.py` | `StandardScaler`, `MinMaxScaler`, `Imputer`, `OneHotEncoder` | `partial_fit` ✅ |
| `metrics.py` | `StreamingAccuracy`, `StreamingPrecision`, `StreamingRecall`, `StreamingF1`, `StreamingConfusionMatrix`, `StreamingAUC`, `RollingAccuracy` | `update` / `result` / `reset` ✅ |
| `stats.py` | `WelfordStatistics`, `StreamingHistogram`, `StreamingQuantile`, `StreamingStats` | `update_stats` ✅ |
| `stream.py` | `StreamTrainer` | `fit_chunk`, `score_chunk` ✅ |
| `pipeline.py` | `Pipeline` | `partial_fit` ✅ |
| `visualise.py` | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth` | — |
| `io.py` | `CSVReader` | `read_chunked` ✅ |

---

## Notes

- **No scikit-learn, pandas, or other ML libraries** are used anywhere. Only NumPy and matplotlib.
- `EnsembleClassifier` is an alias for `RandomForestClassifier` to match the assignment spec naming.
- All `partial_fit` methods on tree-based models use an accumulate-and-rebuild strategy (stores all seen data, rebuilds the tree each call). This produces exact batch-equivalent results.
- The `auc()` function uses `getattr(np, "trapezoid", np.trapz)` for compatibility with both NumPy 1.x and 2.x.
- `StreamingAUC` buffers all `(score, label)` pairs because AUC is not decomposable across chunks.