import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
    plot_confusion_matrix,
    extract_metric_from_logs,
)


def test_plot_metric_over_time_returns_figure():
    fig = plot_metric_over_time(
        [0.6, 0.7, 0.8],
        title="Accuracy over time",
        ylabel="Accuracy",
        show=False,
    )

    assert fig is not None
    assert len(fig.axes) == 1


def test_compare_models_returns_figure():
    fig = compare_models(
        [0.6, 0.7, 0.8],
        [0.5, 0.75, 0.85],
        labels=("Tree", "Forest"),
        show=False,
    )

    assert fig is not None
    assert len(fig.axes) == 1


def test_predictions_vs_ground_truth_returns_figure():
    fig = plot_predictions_vs_ground_truth(
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
        show=False,
    )

    assert fig is not None
    assert len(fig.axes) == 1


def test_confusion_matrix_returns_figure():
    cm = np.array([[2, 1], [0, 3]])

    fig = plot_confusion_matrix(
        cm,
        labels=np.array([0, 1]),
        show=False,
    )

    assert fig is not None
    assert len(fig.axes) >= 1


def test_extract_metric_from_logs():
    logs = [
        {"chunk": 1, "cumulative_accuracy": 0.6},
        {"chunk": 2, "cumulative_accuracy": 0.7},
        {"chunk": 3, "cumulative_accuracy": 0.8},
    ]

    values = extract_metric_from_logs(logs, key="cumulative_accuracy")

    assert np.allclose(values, np.array([0.6, 0.7, 0.8]))


def test_plot_metric_over_time_rejects_empty_input():
    with pytest.raises(ValueError):
        plot_metric_over_time([], show=False)


def test_compare_models_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compare_models([0.6, 0.7], [0.5], show=False)


def test_compare_models_rejects_bad_labels():
    with pytest.raises(ValueError):
        compare_models([0.6, 0.7], [0.5, 0.8], labels=("Only one",), show=False)


def test_predictions_vs_ground_truth_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth(
            np.array([0, 1]),
            np.array([0]),
            show=False,
        )


def test_confusion_matrix_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        plot_confusion_matrix(np.array([[1, 2, 3], [4, 5, 6]]), show=False)


def test_confusion_matrix_rejects_wrong_label_count():
    with pytest.raises(ValueError):
        plot_confusion_matrix(
            np.array([[1, 0], [0, 1]]),
            labels=np.array([0, 1, 2]),
            show=False,
        )


def test_extract_metric_from_logs_rejects_empty_logs():
    with pytest.raises(ValueError):
        extract_metric_from_logs([], key="cumulative_accuracy")


def test_extract_metric_from_logs_rejects_missing_key():
    logs = [{"chunk": 1, "accuracy": 0.8}]

    with pytest.raises(KeyError):
        extract_metric_from_logs(logs, key="cumulative_accuracy")