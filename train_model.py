import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

from unet import efficientB7_unet_builder

seed = 42
np.random.seed = seed
tf.random.set_seed(seed)


main_path = ""


# loading data
training_images = f"{main_path}/images/train/"
training_labels = f"{main_path}/labels/train/"

train_img = next(os.walk(training_images))[2]
train_lbs = next(os.walk(training_labels))[2]

train_img.sort()
train_lbs.sort()

x_train = np.concatenate([np.load(training_images + file_id)["arr_0"] for file_id in train_img], axis=0)
y_train = np.concatenate([np.load(training_labels + file_id)["arr_0"] for file_id in train_lbs], axis=0)

x_train = x_train / 255
y_train = y_train.astype("float64")  # / 255


_, h, w, c = x_train.shape
input_shape = (h, w, c)


# callbacks
early_stop = EarlyStopping(monitor="loss", patience=10)
checkpoint = ModelCheckpoint(filepath=f"{main_path}/checkpoint/", monitor="loss", save_best_only=True, save_freq="epoch")
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.3, patience=5, verbose=1, cooldown=10, min_lr=1e-6)


def scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * tf.math.exp(-0.01)


lr_shceduler = LearningRateScheduler(scheduler, verbose=1)

# training model
model = efficientB7_unet_builder(input_shape)
model.summary()

model.fit(x=x_train, y=y_train, batch_size=16, epochs=500, verbose=2, callbacks=[early_stop, checkpoint, reduce_lr, lr_shceduler])
