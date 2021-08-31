import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Concatenate, Conv2D, Conv2DTranspose, Input, MaxPool2D
from tensorflow.keras.optimizers import Adam

from blocks import Conv_block


def unet_block(x, filters: int, a=0.01, dr=0.05):
    """
    U-net functional block

    Edit this block as needed

    Args:
        x: Input tensor.
        filters (int): No. of filters in convolution layer.
        a (float, optional): Leakage rate for ReLU. Defaults to 0.01.
        dr (float, optional): Dropout rate. Defaults to 0.05.

    Returns:
        Output tensor
    """
    y = Conv_block(x, filters, a, dr)
    # y = Residual_block(x, filters, a, dr, depth=2)
    # y = Recurrent_block(x, filters, a, dr, depth=2)
    # y = R2_block(x, filters, a, dr, depth=2)
    # y = Dense_block(x, filters, a, dr, depth=2)
    # y = Fractal(x, filters, 4, a, dr, depth=2)

    return y


def unet_builder(input_shape: tuple, filters=[32, 32, 48, 80, 224], a=0.01, dr=0.05):
    """
    Generalized U-net builder

    Args:
        input_shape (tuple): Input shape of data.
        filters (list, optional): Filter size per U-net level. Defaults to [32, 32, 48, 80, 224].
        a (float, optional): Leakage rate for ReLU. Defaults to 0.01.
        dr (float, optional): Dropout rate. Defaults to 0.05.

    Returns:
        Keras model
    """
    level = len(filters) - 1
    contract = list()

    # input
    start = Input(input_shape)
    xx = unet_block(start, filters[0], a, dr)

    # contracting path
    for ii in range(level):
        contract.append(xx)
        en = MaxPool2D()(xx)
        xx = unet_block(en, filters[ii + 1], a, dr)

    # expansive path
    for jj in range(level):
        ex = Conv2DTranspose(filters[level - jj], (2, 2), strides=(2, 2), padding="same")(xx)
        ex = Concatenate(axis=-1)([ex, contract[-jj - 1]])
        xx = unet_block(ex, filters[-jj - 2], a, dr)

    # output
    end = Conv2D(1, (1, 1), activation="sigmoid")(xx)

    model = tf.keras.Model(inputs=[start], outputs=[end])

    model.compile(optimizer=Adam(0.01), loss="binary_crossentropy", metrics=["acc", "mse"])

    return model


def efficientB7_unet_builder(input_shape: tuple, filters=[32, 32, 48, 80, 224], a=0.01, dr=0.05):
    """
    EfficientB7 U-net builder

    Args:
        input_shape (tuple): Input shape of data.
        filters (list, optional): Filter size per U-net level. Defaults to [32, 32, 48, 80, 224].
        a (float, optional): Leakage rate for ReLU. Defaults to 0.01.
        dr (float, optional): Dropout rate. Defaults to 0.05.

    Returns:
        [type]: [description]
    """
    level = len(filters) - 1
    contract = list()

    Encoder = EfficientNetB7(include_top=False, weights=None, input_shape=input_shape)

    # input
    start = Encoder.input
    xx = unet_block(start, filters[0], a, dr)

    # contracting path
    for ii in (52, 156, 260, 557):
        contract.append(xx)
        xx = Encoder.layers[ii].output

    # expansive path
    for jj in range(level):
        ex = Conv2DTranspose(filters[level - jj], (2, 2), strides=(2, 2), padding="same")(xx)
        ex = Concatenate(axis=-1)([ex, contract[-jj - 1]])
        xx = unet_block(ex, filters[-jj - 2], a, dr)

    # output
    end = Conv2D(1, (1, 1), activation="sigmoid")(xx)

    model = tf.keras.Model(inputs=[start], outputs=[end])

    model.compile(optimizer=Adam(0.01), loss="binary_crossentropy", metrics=["acc", "mse"])

    return model
