
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize

def metrics_multiclass(y_true, y_pred, y_proba=None, n_classes=None, target_names=None):
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
    if y_proba is not None and n_classes is not None:
        classes = np.arange(n_classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        try:
            auc_macro = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        except ValueError:
            auc_macro = np.nan
        out["roc_auc_macro_ovr"] = auc_macro

    print("\n=== Resumen métricas ===")
    for k, v in out.items():
        if k in ("confusion_matrix", "classification_report"):
            continue
        print(f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}")
    print("\nMatriz de confusión:\n", out["confusion_matrix"])
    print("\nReporte por clase:\n", out["classification_report"])
    return