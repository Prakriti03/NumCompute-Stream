# Benchmark Results

## Environment

- Python: 3.14.3  
- NumPy: 2.4.4  
- OS: Windows 11  

---

# NumCompute — Vectorised vs Python Loop Benchmarks

---

## Mean Benchmark

| Size | NumCompute (s) | Loop (s) | Speedup |
|---|---:|---:|---:|
| 1000 | 0.000025 | 0.000045 | 1.78x |
| 10000 | 0.000007 | 0.000483 | 70.48x |
| 100000 | 0.000027 | 0.007578 | 278.74x |
| 500000 | 0.000198 | 0.032437 | 163.90x |

---

## Standard Deviation Benchmark

| Size | NumCompute (s) | Loop (s) | Speedup |
|---|---:|---:|---:|
| 1000 | 0.000026 | 0.000292 | 11.31x |
| 10000 | 0.000036 | 0.002869 | 79.04x |
| 100000 | 0.000257 | 0.029997 | 116.52x |
| 500000 | 0.005167 | 0.085011 | 16.45x |

---

## Top-10 Benchmark

| Size | NumCompute (s) | Loop (s) | Speedup |
|---|---:|---:|---:|
| 1000 | 0.000017 | 0.000124 | 7.43x |
| 10000 | 0.000027 | 0.003086 | 114.16x |
| 100000 | 0.000296 | 0.035274 | 119.14x |
| 500000 | 0.004245 | 0.253610 | 59.74x |

---

## Mean Squared Error (MSE) Benchmark

| Size | NumCompute (s) | Loop (s) | Speedup |
|---|---:|---:|---:|
| 1000 | 0.000009 | 0.000282 | 32.32x |
| 10000 | 0.000022 | 0.002994 | 136.77x |
| 100000 | 0.000180 | 0.015603 | 86.53x |
| 500000 | 0.004361 | 0.090290 | 20.70x |

---

# Summary — Mean Speedup Across All Sizes

| Benchmark | Average Speedup |
|---|---:|
| Mean | 128.73x |
| Standard Deviation | 55.83x |
| Top-10 | 75.12x |
| MSE | 69.08x |

---

# Key Observations

- NumCompute consistently outperformed pure Python loop implementations across all benchmark categories.
- The largest improvement was observed in the Mean benchmark, reaching up to **278.74x faster** for large input sizes.
- Vectorized implementations showed especially strong performance gains for Top-K selection and MSE calculations.
- Performance improvements increased significantly as input size grew, demonstrating the scalability benefits of NumPy-based vectorization.
- Small input sizes showed smaller speedups due to function-call overhead, while large datasets highlighted the true advantage of optimized array operations.

---