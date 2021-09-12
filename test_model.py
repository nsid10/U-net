import os

import numpy as np
import tensorflow as tf

from metrics import dice_coef, image_metrics

main_path = ""


# loading test data
testing_images = f"{main_path}/images/test/"
test_img = next(os.walk(testing_images))[2]
test_img.sort()
x_test = np.concatenate([np.load(testing_images + file_id)["arr_0"] for file_id in test_img], axis=0)
x_test = x_test / 255

# loading test labels
testing_labels = f"{main_path}/labels/test/"
test_lbl = next(os.walk(testing_labels))[2]
test_lbl.sort()
y_true = np.concatenate([np.load(testing_labels + file_id)["arr_0"] for file_id in test_lbl], axis=0)
y_true = y_true.astype("float64")  # / 255
y_true = y_true.flatten()

# testing model
model = tf.keras.models.load_model(f"{main_path}/checkpoint")
y_pred = model.predict(x_test)
y_pred = y_pred.flatten()

# metrics
dice = dice_coef(y_true, y_pred)
auc, f1, acc, sen, spe, jac = image_metrics(y_true, y_pred, lim=0.5)

print(
    f"Accuracy\t=\t{acc}\nF1 score\t=\t{f1}\nAUC\t\t=\t{auc}\nDice\t\t=\t{dice}\nSensitivity\t=\t{sen}\nSpecificity\t=\t{spe}\nJaccard Index\t=\t{jac}"
)
