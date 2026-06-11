"""
visualise.py — matplotlib plotting helpers for NumCompute-Stream.

The functions in this module are intentionally lightweight and reusable across:

- demo notebooks
- benchmark scripts
- StreamTrainer logs
- report figure generation

Each plotting function returns a matplotlib Figure object and supports both
inline display through show=True and file export through save_path.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _as_1d_array(values, name="values"):
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return arr


def _finalize_plot(title=None, xlabel=None, ylabel=None, save_path=None, show=True):
    """
    Apply common labels/layout, optionally save, optionally display, and return fig.
    """
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig = plt.gcf()  
    
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_metric_over_time(
    metric_values,
    title="Metric over time",
    ylabel="Metric",
    xlabel="Chunk",
    save_path=None,
    show=True,
):
    """
    Plot metric values across streaming chunks.

    Parameters
    ----------
    metric_values : array-like
        Metric value per chunk.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    xlabel : str
        X-axis label.
    save_path : str or None
        Optional path to save figure.
    show : bool
        Whether to display the plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    values = _as_1d_array(metric_values, "metric_values")
    chunks = np.arange(1, values.size + 1)

    plt.figure()
    plt.plot(chunks, values, marker="o")
    return _finalize_plot(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
        show=show,
    )


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    title="Model comparison",
    ylabel="Metric",
    xlabel="Chunk",
    save_path=None,
    show=True,
):
    """
    Compare two models using metric values over streaming chunks.

    Parameters
    ----------
    metric1, metric2 : array-like
        Metric values from two models.
    labels : tuple[str, str]
        Names of the two models.
    """
    values1 = _as_1d_array(metric1, "metric1")
    values2 = _as_1d_array(metric2, "metric2")

    if values1.size != values2.size:
        raise ValueError("metric1 and metric2 must have the same length.")

    if len(labels) != 2:
        raise ValueError("labels must contain exactly two model names.")

    chunks = np.arange(1, values1.size + 1)

    plt.figure()
    plt.plot(chunks, values1, marker="o", label=labels[0])
    plt.plot(chunks, values2, marker="s", label=labels[1])
    plt.legend()

    return _finalize_plot(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
        show=show,
    )


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title="Predictions vs ground truth",
    xlabel="Sample index",
    ylabel="Class label",
    save_path=None,
    show=True,
):
    """
    Plot predicted labels against true labels for one chunk.

    This is useful in the demo to inspect the latest streamed batch and show
    where predictions differ from ground truth.
    """
    y_true = _as_1d_array(y_true, "y_true")
    y_pred = _as_1d_array(y_pred, "y_pred")

    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length.")

    idx = np.arange(y_true.size)

    plt.figure()
    plt.plot(idx, y_true, marker="o", linestyle="-", label="Ground truth")
    plt.plot(idx, y_pred, marker="x", linestyle="--", label="Prediction")
    plt.legend()

    return _finalize_plot(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
        show=show,
    )


def plot_confusion_matrix(
    cm,
    labels=None,
    title="Confusion matrix",
    save_path=None,
    show=True,
):
    """
    Plot a confusion matrix as a heatmap.

    This is not explicitly required, but useful for demo/report.
    """
    cm = np.asarray(cm)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square 2D matrix.")

    if labels is None:
        labels = np.arange(cm.shape[0])
    else:
        labels = np.asarray(labels)
        if labels.size != cm.shape[0]:
            raise ValueError("labels length must match confusion matrix size.")

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(labels.size), labels)
    plt.yticks(np.arange(labels.size), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return plt.gcf()


def extract_metric_from_logs(logs, key="cumulative_accuracy"):
    """
    Extract one scalar field from StreamTrainer logs.

    Parameters
    ----------
    logs : list[dict]
        Log entries returned by StreamTrainer.fit_chunk(), score_chunk(),
        or fit_stream().
    key : str, default="cumulative_accuracy"
        Name of the scalar field to extract.

    Returns
    -------
    np.ndarray
        Metric history across chunks.
    """
    if len(logs) == 0:
        raise ValueError("logs must not be empty.")

    values = []

    for entry in logs:
        if key not in entry:
            raise KeyError(f"Key {key!r} not found in log entry.")
        values.append(entry[key])

    return np.asarray(values, dtype=float)