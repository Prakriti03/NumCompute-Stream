# Cognitive_Coders
Assignment 2.1: Programming Task 1

# NumCompute

NumCompute is a lightweight NumPy-based machine learning utilities library built for numerical computing, preprocessing, statistics, optimization, evaluation metrics, and benchmarking.

The package focuses on implementing commonly used ML and data science utilities from scratch using vectorized NumPy operations while maintaining clean API design, strong test coverage, and performance benchmarking against Python loop baselines.

---

# Features

## Core Modules

### 1. I/O (`io.py`)
- CSV file reader
- Chunked / streaming CSV reading
- Custom delimiter support
- Missing value handling (`NA → np.nan`)
- Automatic dtype conversion

---

### 2. Preprocessing (`preprocessing.py`)
- `StandardScaler`
- `MinMaxScaler`
- `Imputer`
- `OneHotEncoder`

Supports:
- normalization
- standardization
- missing value imputation
- categorical encoding

---

### 3. Sort and Search (`sort_search.py`)
- `sort`
- `argsort`
- `topk`
- `quickselect`
- `binary_search`
- `multikey_sort`

Supports:
- stable sorting
- efficient top-k selection
- binary search insertion logic
- multi-column sorting

---

### 4. Ranking (`rank.py`)
- ranking with tie handling
- percentile computation

Ranking methods:
- average
- dense
- ordinal

---

### 5. Statistics (`stats.py`)
- mean
- median
- variance
- standard deviation
- histogram
- quantile
- percentile

Includes:
- NaN-safe computations
- axis-wise operations
- streaming statistics support

---

### 6. Metrics (`metrics.py`)
Classification:
- accuracy
- precision
- recall
- F1-score
- confusion matrix
- ROC curve
- AUC

Regression:
- MSE

---

### 7. Optimization (`optim.py`)
Numerical differentiation using finite differences:
- gradient estimation
- Jacobian estimation

Supports:
- central difference
- forward difference

Includes:
- input validation
- scalar output validation
- method validation

---

### 8. Pipeline (`pipeline.py`)
Scikit-learn inspired:
- `Transformer`
- `Estimator`
- `Pipeline`

Supports:
- fit
- transform
- predict
- chained preprocessing pipelines

---

### 9. Utilities (`utils.py`)
Distance Functions:
- Euclidean
- Manhattan
- Cosine
- Minkowski

Activation Functions:
- ReLU
- Sigmoid
- Tanh
- Leaky ReLU
- Softmax

Helpers:
- logsumexp
- log_softmax
- gradient clipping
- batching utilities
- one-hot encoding
- pairwise distances

---

### 10. Benchmarking (`benchmarking.py`)
Performance comparison between:
- NumPy vectorized implementations
- Python loop baselines

Benchmarks:
- mean
- std
- top-k
- MSE

Includes:
- speedup analysis
- timing summaries
- scaling benchmarks

---

# Installation

## Clone the repository

```bash
git clone <repository-link>

cd NumCompute

pip install -e .
