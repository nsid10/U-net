import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

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
early_stop = EarlyStopping(monitor="val_loss", patience=15)
checkpoint = ModelCheckpoint(filepath=f"{main_path}/checkpoint", monitor="val_loss", save_best_only=True, save_freq="epoch")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, verbose=1, cooldown=5, min_lr=1e-5)
lr_shceduler = LearningRateScheduler(lambda _, lr: lr * np.exp(-0.01), verbose=1)
tensorboard = TensorBoard(f"{main_path}/logs", update_freq=1)


# training model
model = efficientB7_unet_builder(input_shape)
model.summary()

model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=250,
    verbose=2,
    callbacks=[early_stop, checkpoint, reduce_lr, lr_shceduler, tensorboard],
    validation_split=0.1,
)
