
from .io import CSVReader

from .stats import (
    mean,
    median, 
    std, 
    variance, 
    quantile, 
    percentile, 
    histogram,
    WelfordStatistics,
    StreamingHistogram,
    StreamingQuantile,
    StreamingStats,)

from .preprocessing import StandardScaler, MinMaxScaler, Imputer, OneHotEncoder

from .metrics import (
    accuracy,
    precision,
    recall,
    f1,
    confusion_matrix,
    mse,
    rmse,
    mae,
    r2,
    roc_curve,
    auc,
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingConfusionMatrix,
    RollingAccuracy,
)


from .ensemble import RandomForestClassifier, EnsembleClassifier
from .pipeline import Pipeline
from .stream import StreamTrainer

from .tree import DecisionTreeClassifier, TreeNode

from .utils import (
    euclidean_distance,
    manhattan_distance,
    cosine_similarity,
    softmax,
    relu,
)

from .visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
    plot_confusion_matrix,
    extract_metric_from_logs,
)

__all__ = [
    # io
    "CSVReader",
    
    # stats
    "mean",
    "median",
    "std",
    "variance",
    "quantile",
    "percentile",
    "histogram",
    "WelfordStatistics",
    "StreamingHistogram",
    "StreamingQuantile",
    "StreamingStats",
    
    # metrics
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",
    "rmse",
    "mae",
    "r2",
    "roc_curve",
    "auc",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingConfusionMatrix",
    "RollingAccuracy",

    # preprocessing
    "StandardScaler", "MinMaxScaler", "Imputer", "OneHotEncoder",

    # utils
    "euclidean_distance", "manhattan_distance",
    "cosine_similarity", "softmax", "relu",

    #tree
    "DecisionTreeClassifier",
    "TreeNode",
    
    # ensemble
    "RandomForestClassifier",
    "EnsembleClassifier",

    # pipeline
    "Pipeline",
    
    # stream
    "StreamTrainer",
    
    # visualisation
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
    "plot_confusion_matrix",
    "extract_metric_from_logs",
]