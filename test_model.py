import os

import numpy as np
import tensorflow as tf


main_path = "./"


# loading test data
testing_images = f"{main_path}/Data/images/test/"
test_img = next(os.walk(testing_images))[2]
test_img.sort()
X_test = np.concatenate([np.load(testing_images + file_id)["arr_0"] for file_id in test_img], axis=0)
X_test = X_test / 255

# testing model
model = tf.keras.models.load_model(f"{main_path}/checkpoint/")
preds = model.predict(X_test)

# saving predictions
np.savez_compressed(f"{main_path}/Data/predictions/preds.npz", preds)
