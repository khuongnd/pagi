from sklearn.metrics import *
import numpy as np


def get_binary_classification_metrics(y_true, y_pred):
    y_pred = np.asarray(y_pred)
    y_pred_int = np.where(y_pred >= 0.5, 1, 0)
    return {
        "accuracy": accuracy_score(y_true, y_pred_int),
        "precision": precision_score(y_true, y_pred_int),
        "recall": recall_score(y_true, y_pred_int),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred_int),
        "mcc": matthews_corrcoef(y_true, y_pred_int)
    }
