import numpy as np
from numcompute.metrics import (
    accuracy, precision, recall, f1,
    confusion_matrix, mse, roc_curve, auc
)


# =========================
# 1. Classification Metrics (basic correctness)
# =========================
def test_classification_basic_metrics():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    assert np.isclose(accuracy(y_true, y_pred), 0.8)

    # TP=2, FP=0, FN=1 → precision = 1.0
    assert np.isclose(precision(y_true, y_pred), 1.0)

    # recall = TP/(TP+FN) = 2/3
    assert np.isclose(recall(y_true, y_pred), 2/3)

    # f1 should be harmonic mean
    f1_val = f1(y_true, y_pred)
    assert 0 <= f1_val <= 1


# =========================
# 2. Confusion Matrix correctness
# =========================
def test_confusion_matrix_structure():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 2])

    cm, labels = confusion_matrix(y_true, y_pred)

    assert cm.shape == (3, 3)

    # Known structure from docstring example
    expected = np.array([
        [2, 0, 0],
        [1, 0, 1],
        [0, 1, 1]
    ])

    assert np.array_equal(cm, expected)
    assert len(labels) == 3


# =========================
# 3. Regression metric (MSE correctness)
# =========================
def test_mse_and_numerical_stability():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    # (0^2 + 0^2 + 1^2)/3 = 1/3
    assert np.isclose(mse(y_true, y_pred), 1/3)

    # edge case: perfect prediction
    assert mse(y_true, y_true) == 0.0


