import os

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


main_path = ""


# loading test data
testing_labels = f"{main_path}/labels/test/"
test_lbl = next(os.walk(testing_labels))[2]
test_lbl.sort()
y_true = np.concatenate([np.load(testing_labels + file_id)["arr_0"] for file_id in test_lbl], axis=0)
y_true = y_true.astype("float64")  # / 255
y_true = y_true.flatten()


# loading predictions
y_pred = np.load(f"{main_path}/predictions/preds.npz")["arr_0"]
y_pred = y_pred.flatten()


def dice_coef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Sørensen–Dice coefficient
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    return (2.0 * np.sum(y_true_f * y_pred_f)) / (np.sum(y_true_f) + np.sum(y_pred_f))


def image_metrics(y_true: np.ndarray, y_pred: np.ndarray, lim=0.5) -> tuple(float):
    """
    Calculates Area Under the Curve, F-score, Accuracy, Sensitivity, and Specificity
    """
    Y_pred_bin = np.zeros_like(y_true)
    idx = y_pred > lim
    Y_pred_bin[idx] = 1

    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, Y_pred_bin)
    acc = accuracy_score(y_true, Y_pred_bin)
    mat = confusion_matrix(y_true, Y_pred_bin)

    TN = mat[0][0]
    FN = mat[1][0]
    TP = mat[1][1]
    FP = mat[0][1]

    sen = TP / (TP + FN)
    spe = TN / (TN + FP)

    return auc, f1, acc, sen, spe


dice = dice_coef(y_true, y_pred)
auc, f1, acc, sen, spe = image_metrics(y_true, y_pred, lim=0.5)


print(f"Accuracy = \t{acc}\nf1score\t = \t{f1}\nAUC\t = \t{auc}\nDice\t = \t{dice}\nSensitivity = \t{sen}\nSpecificity = \t{spe}")
