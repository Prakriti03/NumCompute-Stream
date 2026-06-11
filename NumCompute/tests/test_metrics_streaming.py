"""
Unit tests for streaming metrics.

Coverage includes:
- cumulative accuracy
- accumulating confusion matrix
- precision, recall, and F1 from streamed batches
- rolling-window accuracy
- streaming AUC accumulation
- reset behaviour
"""
import numpy as np

from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingConfusionMatrix,
    RollingAccuracy,
    StreamingAUC,
)


def test_streaming_accuracy_updates_across_chunks():
    metric = StreamingAccuracy()

    metric.update(np.array([1, 0]), np.array([1, 1]))
    metric.update(np.array([1, 1]), np.array([1, 0]))

    assert np.isclose(metric.result(), 0.5)


def test_streaming_confusion_matrix_accumulates():
    metric = StreamingConfusionMatrix(labels=np.array([0, 1]))

    metric.update(np.array([0, 1]), np.array([0, 1]))
    metric.update(np.array([0, 1]), np.array([1, 1]))

    cm, labels = metric.result()

    assert cm.shape == (2, 2)
    assert cm.sum() == 4
    assert np.array_equal(labels, np.array([0, 1]))


def test_streaming_precision_recall_f1_binary():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])

    precision = StreamingPrecision().update(y_true, y_pred).result()
    recall = StreamingRecall().update(y_true, y_pred).result()
    f1 = StreamingF1().update(y_true, y_pred).result()

    assert np.isclose(precision, 0.5)
    assert np.isclose(recall, 0.5)
    assert np.isclose(f1, 0.5)


def test_rolling_accuracy_uses_recent_window_only():
    metric = RollingAccuracy(window_size=3)

    metric.update(np.array([1, 1, 1]), np.array([1, 0, 0]))
    metric.update(np.array([1, 1]), np.array([1, 1]))

    assert np.isclose(metric.result(), 2 / 3)


def test_reset_clears_metric_state():
    metric = StreamingAccuracy()

    metric.update(np.array([1, 0]), np.array([1, 1]))
    metric.reset()

    assert metric.result() == 0.0

def test_streaming_auc_accumulates_scores_across_chunks():
    metric = StreamingAUC()

    metric.update(np.array([0, 1]), np.array([0.1, 0.8]))
    metric.update(np.array([0, 1]), np.array([0.2, 0.9]))

    assert np.isclose(metric.result(), 1.0)
    
def test_confusion_matrix_expands_when_new_class_appears():
    metric = StreamingConfusionMatrix()

    metric.update(np.array([0, 1]), np.array([0, 1]))
    metric.update(np.array([2, 2]), np.array([2, 1]))

    cm, labels = metric.result()

    assert cm.shape == (3, 3)
    assert np.array_equal(labels, np.array([0, 1, 2]))
    assert cm.sum() == 4