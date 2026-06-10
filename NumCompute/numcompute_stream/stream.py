"""
stream.py — streaming training utilities.

Provides StreamTrainer for chunk-wise training, scoring, metric tracking,
memory logging, and cumulative accuracy monitoring.
"""

from __future__ import annotations

import sys
import time
import numpy as np

class StreamTrainer:
    """
    Manage chunk-wise training and evaluation for streaming ML workflows.

    Parameters
    ----------
    pipeline : object
        Pipeline or model with partial_fit(), predict(), and optionally score().
    metrics : dict or None
        Optional dictionary of metric objects with update(), result(), reset().
    track_memory : bool
        Whether to estimate memory footprint per chunk.
    """

    def __init__(self, pipeline, metrics=None, track_memory=True):
        if not hasattr(pipeline, "partial_fit"):
            raise TypeError("pipeline must implement partial_fit().")
        if not hasattr(pipeline, "predict"):
            raise TypeError("pipeline must implement predict().")

        self.pipeline = pipeline
        self.metrics = metrics if metrics is not None else {}
        self.track_memory = track_memory

        self.logs_ = []
        self.chunk_index_ = 0
        self.total_correct_ = 0
        self.total_seen_ = 0
        self._is_fitted = False
                # Add to __init__:
        self.total_eval_correct_ = 0
        self.total_eval_seen_    = 0

    def _validate_chunk(self, X, y=None):
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        if X.shape[0] == 0:
            raise ValueError("X chunk must contain at least one sample.")

        if y is not None:
            y = np.asarray(y)

            if y.ndim != 1:
                raise ValueError(f"y must be 1D, got shape {y.shape}")

            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}"
                )

        return X, y

    def _estimate_memory_bytes(self, X, y=None):
        total = 0

        if hasattr(X, "nbytes"):
            total += X.nbytes
        else:
            total += sys.getsizeof(X)

        if y is not None:
            if hasattr(y, "nbytes"):
                total += y.nbytes
            else:
                total += sys.getsizeof(y)

        return int(total)

    def _update_metrics(self, y_true, y_pred):
        for metric in self.metrics.values():
            if not hasattr(metric, "update"):
                raise TypeError("Each metric object must implement update().")
            metric.update(y_true, y_pred)

    def _metric_results(self):
        results = {}

        for name, metric in self.metrics.items():
            if not hasattr(metric, "result"):
                raise TypeError("Each metric object must implement result().")
            results[name] = metric.result()

        return results

    def fit_chunk(self, X, y, classes=None):
        """
        Train the pipeline/model on one incoming chunk.

        Returns
        -------
        dict
            Log entry containing chunk number, size, training time,
            memory footprint, chunk accuracy, cumulative accuracy, and metrics.
        """
        X, y = self._validate_chunk(X, y)

        start = time.perf_counter()

        try:
            self.pipeline.partial_fit(X, y, classes=classes)
        except TypeError:
            self.pipeline.partial_fit(X, y)

        y_pred = self.pipeline.predict(X)

        elapsed = time.perf_counter() - start

        chunk_correct = int(np.sum(y_pred == y))
        chunk_size = int(y.shape[0])
        chunk_accuracy = float(chunk_correct / chunk_size)

        self.total_correct_ += chunk_correct
        self.total_seen_ += chunk_size
        cumulative_accuracy = float(self.total_correct_ / self.total_seen_)

        self._update_metrics(y, y_pred)

        memory_bytes = (
            self._estimate_memory_bytes(X, y)
            if self.track_memory
            else None
        )

        self.chunk_index_ += 1
        self._is_fitted = True

        log_entry = {
            "chunk": self.chunk_index_,
            "chunk_size": chunk_size,
            "train_time_sec": float(elapsed),
            "memory_bytes": memory_bytes,
            "chunk_accuracy": chunk_accuracy,
            "cumulative_accuracy": cumulative_accuracy,
            "metrics": self._metric_results(),
        }

        self.logs_.append(log_entry)

        return log_entry

    def score_chunk(self, X, y):
        """
        Score the current pipeline/model on one chunk without training.

        Returns
        -------
        dict
            Log entry containing chunk accuracy, cumulative accuracy,
            memory footprint, and metric results.
        """
        if not self._is_fitted:
            raise RuntimeError("StreamTrainer must be fitted before score_chunk().")

        X, y = self._validate_chunk(X, y)

        start = time.perf_counter()
        y_pred = self.pipeline.predict(X)
        elapsed = time.perf_counter() - start

        chunk_correct = int(np.sum(y_pred == y))
        chunk_size = int(y.shape[0])
        chunk_accuracy = float(chunk_correct / chunk_size)

        self.total_eval_correct_ += chunk_correct
        self.total_eval_seen_    += chunk_size
        cumulative_accuracy = float(self.total_eval_correct_ / self.total_eval_seen_)

        self._update_metrics(y, y_pred)

        memory_bytes = (
            self._estimate_memory_bytes(X, y)
            if self.track_memory
            else None
        )

        self.chunk_index_ += 1

        log_entry = {
            "chunk": self.chunk_index_,
            "chunk_size": chunk_size,
            "score_time_sec": float(elapsed),
            "memory_bytes": memory_bytes,
            "chunk_accuracy": chunk_accuracy,
            "cumulative_accuracy": cumulative_accuracy,
            "metrics": self._metric_results(),
        }

        self.logs_.append(log_entry)

        return log_entry

    def fit_stream(self, X, y, chunk_size=100, classes=None):
        """
        Fit over a full dataset by splitting it into chunks.

        This is useful for demos where a static dataset is simulated as a stream.
        """
        X, y = self._validate_chunk(X, y)

        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")

        logs = []

        for start in range(0, X.shape[0], chunk_size):
            end = start + chunk_size
            log = self.fit_chunk(X[start:end], y[start:end], classes=classes)
            logs.append(log)

        return logs

    def get_logs(self):
        """Return logs as a list of dictionaries."""
        return list(self.logs_)

    def get_metric_history(self, key="cumulative_accuracy"):
        """Return a NumPy array of a logged scalar value across chunks."""
        return np.asarray([entry[key] for entry in self.logs_], dtype=float)

    def reset(self):
        """Reset trainer logs and counters."""
        self.logs_ = []
        self.chunk_index_ = 0
        self.total_correct_ = 0
        self.total_seen_ = 0
        self._is_fitted = False
        
        self.total_eval_correct_ = 0
        self.total_eval_seen_    = 0   

        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()

        return self

    def __repr__(self):
        return (
            "StreamTrainer("
            f"chunks_seen={self.chunk_index_}, "
            f"total_seen={self.total_seen_}, "
            f"is_fitted={self._is_fitted})"
        )