import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, jaccard_score, roc_auc_score


def dice_coef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Sørensen–Dice coefficient
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    return (2.0 * np.sum(y_true_f * y_pred_f)) / (np.sum(y_true_f) + np.sum(y_pred_f))


def image_metrics(y_true: np.ndarray, y_pred: np.ndarray, lim=0.5):
    """
    Calculates Area Under the Curve, F-score, Accuracy, Sensitivity, Specificity, Jaccard Index
    """
    y_pred_bin = np.zeros_like(y_true)
    idx = y_pred >= lim
    y_pred_bin[idx] = 1

    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_bin)
    acc = accuracy_score(y_true, y_pred_bin)
    mat = confusion_matrix(y_true, y_pred_bin)
    jac = jaccard_score(y_true, y_pred_bin, normalize=True)

    TN = mat[0][0]
    FN = mat[1][0]
    TP = mat[1][1]
    FP = mat[0][1]

    sen = TP / (TP + FN)
    spe = TN / (TN + FP)

    return auc, f1, acc, sen, spe, jac
