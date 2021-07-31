import tensorflow as tf

from blocks import Conv_block

from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Concatenate, Conv2D, Conv2DTranspose, Input, MaxPool2D
from tensorflow.keras.optimizers import Adam


def unet_block(x, filters: int):
    """
    U-net functional block

    Edit this block as needed
    """
    a = 0.05  # Leakage rate for ReLU.
    dr = 0.1  # Dropout rate.

    y = Conv_block(x, filters, a, dr)

    return y


def unet_builder(input_shape: tuple):
    """
    Generalized U-net builder

    Args:
        input_shape (tuple): Input shape of data

    Returns:
        Keras model
    """
    filters = [32, 32, 48, 80, 224]
    level = len(filters) - 1
    contract = list()

    # input
    start = Input(input_shape)
    xx = unet_block(start, filters[0])

    # contracting path
    for ii in range(level):
        contract.append(xx)
        en = MaxPool2D()(xx)
        xx = unet_block(en, filters[ii + 1])

    # expansive path
    for jj in range(level):
        ex = Conv2DTranspose(filters[level - jj], (2, 2), strides=(2, 2), padding="same")(xx)
        ex = Concatenate(axis=-1)([ex, contract[-jj - 1]])
        xx = unet_block(ex, filters[-jj - 2])

    # output
    end = Conv2D(1, (1, 1), activation="sigmoid")(xx)

    model = tf.keras.Model(inputs=[start], outputs=[end])

    model.compile(optimizer=Adam(0.01), loss="binary_crossentropy", metrics=["acc", "mse"])

    return model


def efficientB7_unet_builder(input_shape: tuple):
    """
    EfficientNetB7 U-net builder

    Args:
        input_shape (tuple): Input shape of data

    Returns:
        Keras model
    """
    filters = [32, 32, 48, 80, 224]
    level = len(filters) - 1
    contract = list()

    Encoder = EfficientNetB7(include_top=False, weights=None, input_shape=input_shape)

    # input
    start = Encoder.input
    xx = unet_block(start, filters[0])

    # contracting path
    for ii in (52, 156, 260, 557):
        contract.append(xx)
        xx = Encoder.layers[ii].output

    # expansive path
    for jj in range(level):
        ex = Conv2DTranspose(filters[level - jj], (2, 2), strides=(2, 2), padding="same")(xx)
        ex = Concatenate(axis=-1)([ex, contract[-jj - 1]])
        xx = unet_block(ex, filters[-jj - 2])

    # output
    end = Conv2D(1, (1, 1), activation="sigmoid")(xx)

    model = tf.keras.Model(inputs=[start], outputs=[end])

    model.compile(optimizer=Adam(0.01), loss="binary_crossentropy", metrics=["acc", "mse"])

    return model
