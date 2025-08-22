
"""
Utility functions for evaluating multiclass classification models.
Provides accuracy, balanced accuracy, precision, recall, F1-score,
confusion matrix, classification report, and macro AUC (if probabilities are provided).
"""

from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
import numpy as np
from sklearn.preprocessing import label_binarize

def metrics_multiclass(y_true, y_pred, y_proba=None, n_classes=None, target_names=None):
    """
    Compute and display metrics for multiclass classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth (true class labels).
    y_pred : array-like
        Predicted class labels.
    y_proba : array-like, optional (default=None)
        Predicted probabilities for each class. Required for ROC AUC computation.
    n_classes : int, optional (default=None)
        Number of classes in the problem. Required if y_proba is provided.
    target_names : list of str, optional (default=None)
        Names of target classes to display in the classification report.

    Returns
    -------
    out : dict
        Dictionary containing all computed metrics, including:
        - accuracy
        - balanced_accuracy
        - precision_macro
        - recall_macro
        - f1_macro
        - confusion_matrix
        - classification_report
        - roc_auc_macro_ovr (if y_proba provided)
    """

    out = {
        "accuracy": (y_true == y_pred).mean(),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, target_names=target_names, digits=3, zero_division=0
        )
    }

    #ROC AUC (only if probabilities are provided)
    if y_proba is not None and n_classes is not None:
        classes = np.arange(n_classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        try:
            auc_macro = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        except ValueError:
            auc_macro = np.nan
        out["roc_auc_macro_ovr"] = auc_macro

    print("\n=== Metrics Summary ===")
    for k, v in out.items():
        if k in ("confusion_matrix", "classification_report"):
            continue
        print(f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}")
    print("\nConfusion Matrix:\n", out["confusion_matrix"])
    print("\nClassification Report:\n", out["classification_report"])
    return