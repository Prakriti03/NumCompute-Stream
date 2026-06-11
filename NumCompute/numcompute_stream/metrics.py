"""
metrics.py — batch and streaming evaluation metrics.

Provides standard classification and regression metrics together with
streaming metric classes that support:

- update(y_true_chunk, y_pred_chunk)
- result()
- reset()

Streaming metrics are designed for use with StreamTrainer, where predictions
arrive chunk by chunk. Confusion matrix and AUC accumulate information over
time, while RollingAccuracy keeps only a fixed recent window.
"""

import numpy as np

def _as_1d(arr, name='array'):
    """Convert input to a NumPy array and require one-dimensional shape."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    return arr

def _check_len(*arrays):
    """Raise ValueError if input arrays do not share the same first dimension."""
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"Arrays have inconsistent lengths: {sorted(lengths)}")

def _per_class_counts(y_true, y_pred, labels):
    """Return (TP, FP, FN) arrays — one entry per class — fully vectorised.
    Notes
    -----
    The small label-mapping loop iterates over classes, not samples.
    Counting is vectorised through np.bincount."""
    k = labels.size
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    def _map(arr):
        idx = np.full(arr.shape, -1, dtype=int)
        for lbl, i in label_to_idx.items():
            idx[arr == lbl] = i
        return idx

    ti = _map(y_true);  pi = _map(y_pred)
    valid = (ti >= 0) & (pi >= 0)
    ti, pi = ti[valid], pi[valid]

    tp = np.bincount(ti[ti == pi], minlength=k).astype(float)
    fp = np.bincount(pi, minlength=k).astype(float) - tp
    fn = np.bincount(ti, minlength=k).astype(float) - tp
    return tp, fp, fn

#confusion matrix

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Confusion matrix.

    Returns
    -------
    cm     : np.ndarray, shape (k, k)
             cm[i, j] = # samples with true label i predicted as j
    labels : np.ndarray, shape (k,)

    Examples
    --------
    >>> cm, lbl = confusion_matrix(
    ...     np.array([0,1,2,0,1,2]),
    ...     np.array([0,2,1,0,0,2]))
    >>> cm
    array([[2, 0, 0],
           [1, 0, 1],
           [0, 1, 1]])
    """
    y_true = _as_1d(np.asarray(y_true), 'y_true')
    y_pred = _as_1d(np.asarray(y_pred), 'y_pred')
    _check_len(y_true, y_pred)
    if labels is None:
        labels = np.union1d(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    k = labels.size
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    def _map(arr):
        idx = np.full(arr.shape, -1, dtype=int)
        for lbl, i in label_to_idx.items():
            idx[arr == lbl] = i
        return idx

    ti = _map(y_true);  pi = _map(y_pred)
    valid = (ti >= 0) & (pi >= 0)
    cm = np.bincount(ti[valid] * k + pi[valid], minlength=k*k).reshape(k, k)
    return cm, labels

#accuracy
def accuracy(y_true, y_pred):
    """
    Fraction of correctly predicted labels.

    Examples
    --------
    >>> accuracy(np.array([1,0,1,1]), np.array([1,0,0,1]))
    0.75
    """
    y_true = _as_1d(np.asarray(y_true), 'y_true')
    y_pred = _as_1d(np.asarray(y_pred), 'y_pred')
    _check_len(y_true, y_pred)
    if y_true.size == 0:
        raise ValueError("y_true is empty.")
    return float(np.mean(y_true == y_pred))
# precision / recall / f1
def _clf_metric(metric, y_true, y_pred, average, pos_label, zero_division):
    _OK = {'binary', 'macro', 'micro', 'none'}
    if average not in _OK:
        raise ValueError(f"average={average!r} must be one of {_OK}")

    y_true = _as_1d(np.asarray(y_true), 'y_true')
    y_pred = _as_1d(np.asarray(y_pred), 'y_pred')
    _check_len(y_true, y_pred)

    labels      = np.union1d(y_true, y_pred)
    tp, fp, fn  = _per_class_counts(y_true, y_pred, labels)

    if metric == 'precision':
        denom      = tp + fp
        per_class  = np.where(denom == 0, zero_division, tp / denom)
        micro_num  = tp.sum();  micro_den = (tp + fp).sum()
    elif metric == 'recall':
        denom      = tp + fn
        per_class  = np.where(denom == 0, zero_division, tp / denom)
        micro_num  = tp.sum();  micro_den = (tp + fn).sum()
    else:  # f1
        denom      = 2*tp + fp + fn
        per_class  = np.where(denom == 0, zero_division, 2*tp / denom)
        micro_num  = 2*tp.sum();  micro_den = (2*tp + fp + fn).sum()

    if average == 'none'  : return per_class
    if average == 'macro' : return float(np.mean(per_class))
    if average == 'micro' :
        return float(micro_num / micro_den) if micro_den > 0 else zero_division
    if pos_label not in labels:
        return zero_division
    idx = np.where(labels == pos_label)[0][0]
    return float(per_class[idx])


def precision(y_true, y_pred, average='binary', pos_label=1, zero_division=0.0):
    """
    Precision = TP / (TP + FP).

    Parameters
    ----------
    average : 'binary' | 'macro' | 'micro' | 'none'
    pos_label : positive class for 'binary' mode

    Examples
    --------
    >>> precision(np.array([0,1,1,0,1]), np.array([0,1,0,0,1]))
    0.6666...
    """
    return _clf_metric('precision', y_true, y_pred, average, pos_label, zero_division)


def recall(y_true, y_pred, average='binary', pos_label=1, zero_division=0.0):
    """
    Recall = TP / (TP + FN).

    Examples
    --------
    >>> recall(np.array([0,1,1,0,1]), np.array([0,1,0,0,1]))
    0.6666...
    """
    return _clf_metric('recall', y_true, y_pred, average, pos_label, zero_division)


def f1(y_true, y_pred, average='binary', pos_label=1, zero_division=0.0):
    """
    F1 = 2*TP / (2*TP + FP + FN).

    Examples
    --------
    >>> f1(np.array([0,1,1,0,1]), np.array([0,1,0,0,1]))
    0.6666...
    """
    return _clf_metric('f1', y_true, y_pred, average, pos_label, zero_division)

# regression metrics

def mse(y_true, y_pred):
    """
    Mean Squared Error.

    Examples
    --------
    >>> mse(np.array([1., 2., 3.]), np.array([1., 2., 4.]))
    0.3333...
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Inputs are empty.")
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    """Root Mean Squared Error = sqrt(MSE)."""
    return float(np.sqrt(mse(y_true, y_pred)))
def mae(y_true, y_pred):
    """Mean Absolute Error = mean(|y_true - y_pred|)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.mean(np.abs(y_true - y_pred)))
def r2(y_true, y_pred):
    """
    Coefficient of determination R² = 1 - SS_res / SS_tot.
    Returns 0.0 when y_true is constant.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

# ROC curve + AUC 
def roc_curve(y_true, y_score, pos_label=1):
    """
    ROC curve for binary classification.

    Parameters
    ----------
    y_true   : 1-D array of binary labels
    y_score  : 1-D array of scores (higher → more likely positive)
    pos_label: the positive class label

    Returns
    -------
    fpr        : np.ndarray  (false positive rates)
    tpr        : np.ndarray  (true positive rates / recall)
    thresholds : np.ndarray  (score thresholds in descending order)

    Examples
    --------
    >>> fpr, tpr, th = roc_curve(
    ...     np.array([0,0,1,1]),
    ...     np.array([0.1,0.4,0.35,0.8]))
    """
    y_true  = _as_1d(np.asarray(y_true),          'y_true')
    y_score = _as_1d(np.asarray(y_score, float),  'y_score')
    _check_len(y_true, y_score)

    pos   = (y_true == pos_label).astype(float)
    n_pos = pos.sum();  n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least one positive and one negative sample.")

    order      = np.argsort(-y_score, kind='stable')
    sorted_pos = pos[order]
    thresholds = y_score[order]

    tp_cum = np.cumsum(sorted_pos)
    fp_cum = np.cumsum(1.0 - sorted_pos)

    tpr = np.concatenate(([0.], tp_cum / n_pos))
    fpr = np.concatenate(([0.], fp_cum / n_neg))
    if fpr.size > 2:
        keep = np.ones(fpr.size, dtype=bool)
        keep[1:-1] = (np.diff(fpr[:-1]) != 0) | (np.diff(tpr[:-1]) != 0)
        fpr        = fpr[keep]
        tpr        = tpr[keep]
        thresholds = thresholds[keep[1:]]

    return fpr, tpr, thresholds


def auc(fpr, tpr):
    """
    Area Under the Curve via the trapezoidal rule.

    Examples
    --------
    >>> auc(np.array([0., 0.5, 1.]), np.array([0., 0.5, 1.]))
    0.5
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape.")
    if fpr.size < 2:
        raise ValueError("Need at least 2 points to compute AUC.")
    order = np.argsort(fpr, kind='stable')
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(tpr[order], fpr[order]))

# Streaming Classification Metrics
class StreamingConfusionMatrix:
    """
    Accumulating confusion matrix for streaming classification.

    If labels=None, the label set expands automatically when new classes appear
    in later chunks. Existing counts are reindexed into the expanded matrix.
    """
    def __init__(self, labels=None):
        self._auto = labels is None
        self.labels = None if labels is None else np.asarray(labels)
        self.cm = None
        if self.labels is not None:
            self.cm = np.zeros((self.labels.size, self.labels.size), dtype=int)

    def update(self, y_true, y_pred):
        y_true = _as_1d(np.asarray(y_true), "y_true")
        y_pred = _as_1d(np.asarray(y_pred), "y_pred")
        _check_len(y_true, y_pred)
        if y_true.size == 0:
            raise ValueError("y_true and y_pred must not be empty.")

        if self._auto:
            batch_labels = np.union1d(y_true, y_pred)
            if self.labels is None:
                self.labels = batch_labels
                self.cm = np.zeros((self.labels.size, self.labels.size), dtype=int)
            else:
                all_labels = np.union1d(self.labels, batch_labels)
                if all_labels.size != self.labels.size:
                    old_cm, old_labels = self.cm, self.labels
                    self.labels = all_labels
                    self.cm = np.zeros((all_labels.size, all_labels.size), dtype=int)
                    idx = np.searchsorted(all_labels, old_labels)   # union1d is sorted
                    self.cm[np.ix_(idx, idx)] = old_cm

        batch_cm, _ = confusion_matrix(y_true, y_pred, labels=self.labels)
        self.cm += batch_cm
        return self

    def result(self):
        if self.cm is None:
            return np.zeros((0, 0), dtype=int), np.array([])
        return self.cm.copy(), self.labels.copy()

    def reset(self):
        self.cm = None
        return self


class StreamingAccuracy:
    """Streaming accuracy using cumulative correct and total counts."""
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, y_true, y_pred):
        y_true = _as_1d(np.asarray(y_true), "y_true")
        y_pred = _as_1d(np.asarray(y_pred), "y_pred")
        _check_len(y_true, y_pred)

        if y_true.size == 0:
            raise ValueError("y_true and y_pred must not be empty.")

        self.correct += int(np.sum(y_true == y_pred))
        self.total += int(y_true.size)

        return self

    def result(self):
        return 0.0 if self.total == 0 else float(self.correct / self.total)

    def reset(self):
        self.correct = 0
        self.total = 0
        return self


class StreamingPrecision:
    """Streaming precision computed from an accumulated confusion matrix."""
    def __init__(self, average="binary", pos_label=1, zero_division=0.0, labels=None):
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division
        self.cm_metric = StreamingConfusionMatrix(labels=labels)

    def update(self, y_true, y_pred):
        self.cm_metric.update(y_true, y_pred)
        return self

    def result(self):
        cm, labels = self.cm_metric.result()
        if cm.size == 0:
            return self.zero_division

        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp

        denom = tp + fp
        per_class = np.where(denom == 0, self.zero_division, tp / denom)

        if self.average == "none":
            return per_class

        if self.average == "macro":
            return float(np.mean(per_class))

        if self.average == "micro":
            total_tp = tp.sum()
            total_fp = fp.sum()
            den = total_tp + total_fp
            return float(total_tp / den) if den > 0 else self.zero_division

        if self.average == "binary":
            if self.pos_label not in labels:
                return self.zero_division
            idx = np.where(labels == self.pos_label)[0][0]
            return float(per_class[idx])

        raise ValueError("average must be 'binary', 'macro', 'micro', or 'none'")

    def reset(self):
        self.cm_metric.reset()
        return self


class StreamingRecall:
    """Streaming recall computed from an accumulated confusion matrix."""
    def __init__(self, average="binary", pos_label=1, zero_division=0.0, labels=None):
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division
        self.cm_metric = StreamingConfusionMatrix(labels=labels)

    def update(self, y_true, y_pred):
        self.cm_metric.update(y_true, y_pred)
        return self

    def result(self):
        cm, labels = self.cm_metric.result()
        if cm.size == 0:
            return self.zero_division

        tp = np.diag(cm).astype(float)
        fn = cm.sum(axis=1) - tp

        denom = tp + fn
        per_class = np.where(denom == 0, self.zero_division, tp / denom)

        if self.average == "none":
            return per_class

        if self.average == "macro":
            return float(np.mean(per_class))

        if self.average == "micro":
            total_tp = tp.sum()
            total_fn = fn.sum()
            den = total_tp + total_fn
            return float(total_tp / den) if den > 0 else self.zero_division

        if self.average == "binary":
            if self.pos_label not in labels:
                return self.zero_division
            idx = np.where(labels == self.pos_label)[0][0]
            return float(per_class[idx])

        raise ValueError("average must be 'binary', 'macro', 'micro', or 'none'")

    def reset(self):
        self.cm_metric.reset()
        return self


class StreamingF1:
    """Streaming F1 score computed from an accumulated confusion matrix."""
    def __init__(self, average="binary", pos_label=1, zero_division=0.0, labels=None):
        self.average = average
        self.pos_label = pos_label
        self.zero_division = zero_division
        self.cm_metric = StreamingConfusionMatrix(labels=labels)

    def update(self, y_true, y_pred):
        self.cm_metric.update(y_true, y_pred)
        return self

    def result(self):
        cm, labels = self.cm_metric.result()
        if cm.size == 0:
            return self.zero_division

        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        denom = 2 * tp + fp + fn
        per_class = np.where(denom == 0, self.zero_division, (2 * tp) / denom)

        if self.average == "none":
            return per_class

        if self.average == "macro":
            return float(np.mean(per_class))

        if self.average == "micro":
            total_tp = tp.sum()
            total_fp = fp.sum()
            total_fn = fn.sum()
            den = 2 * total_tp + total_fp + total_fn
            return float((2 * total_tp) / den) if den > 0 else self.zero_division

        if self.average == "binary":
            if self.pos_label not in labels:
                return self.zero_division
            idx = np.where(labels == self.pos_label)[0][0]
            return float(per_class[idx])

        raise ValueError("average must be 'binary', 'macro', 'micro', or 'none'")

    def reset(self):
        self.cm_metric.reset()
        return self


class RollingAccuracy:
    """
    Rolling-window accuracy over the most recent predictions.

    Unlike StreamingAccuracy, this metric only keeps the last window_size
    correctness flags, making it useful for recent performance monitoring.
    """
    def __init__(self, window_size=100):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.window_size = window_size
        self.correct_buffer = np.array([], dtype=int)

    def update(self, y_true, y_pred):
        y_true = _as_1d(np.asarray(y_true), "y_true")
        y_pred = _as_1d(np.asarray(y_pred), "y_pred")
        _check_len(y_true, y_pred)

        if y_true.size == 0:
            raise ValueError("y_true and y_pred must not be empty.")

        correct = (y_true == y_pred).astype(int)

        self.correct_buffer = np.concatenate([self.correct_buffer, correct])

        if self.correct_buffer.size > self.window_size:
            self.correct_buffer = self.correct_buffer[-self.window_size:]

        return self

    def result(self):
        if self.correct_buffer.size == 0:
            return 0.0
        return float(np.mean(self.correct_buffer))

    def reset(self):
        self.correct_buffer = np.array([], dtype=int)
        return self
    
class StreamingAUC:
    """
    Accumulating ROC-AUC for binary streaming classification.

    AUC is not decomposable by simply averaging chunk-level AUC values, so this
    class buffers all labels and scores seen so far and recomputes ROC-AUC in
    result(). Until both positive and negative labels have been observed,
    result() returns 0.0.
    """

    def __init__(self, pos_label=1):
        self.pos_label = pos_label
        self._scores = np.array([], dtype=float)
        self._labels = np.array([])

    def update(self, y_true, y_score):
        y_true = _as_1d(np.asarray(y_true), "y_true")
        y_score = _as_1d(np.asarray(y_score, float), "y_score")
        _check_len(y_true, y_score)
        if y_true.size == 0:
            raise ValueError("y_true and y_score must not be empty.")
        self._labels = np.concatenate([self._labels, y_true])
        self._scores = np.concatenate([self._scores, y_score])
        return self

    def result(self):
        pos = (self._labels == self.pos_label)
        if pos.sum() == 0 or (~pos).sum() == 0:
            return 0.0  # undefined until both classes seen
        fpr, tpr, _ = roc_curve(self._labels, self._scores, self.pos_label)
        return auc(fpr, tpr)

    def reset(self):
        self._scores = np.array([], dtype=float)
        self._labels = np.array([])
        return self