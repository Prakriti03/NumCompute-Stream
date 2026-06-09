import numpy as np

from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingConfusionMatrix,
    RollingAccuracy,
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