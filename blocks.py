import tensorflow as tf

from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, LeakyReLU, MaxPooling2D


def Conv_block(x, filters, a=0.01, dr=0.05, k=3, s=1):
    x = Conv2D(filters, (k, k), strides=(s, s), padding="same")(x)
    x = LeakyReLU(alpha=a)(x)
    x = BatchNormalization(axis=-1)(x)
    y = Dropout(rate=dr)(y)

    return y


def Residual_block(x, filters, a=0.01, dr=0.05):
    y = Conv_block(x, filters, a, dr)
    y = Conv_block(y, filters, a, dr)
    y = Add(axis=-1)([x, y])

    return y


def Dense_block(x, filters, a=0.01, dr=0.05, depth=2):

    for _ in range(depth):
        xn = Conv_block(x, filters, a, dr)
        x = Concatenate(axis=-1)([x, xn])

    return x


def R2_block_old(x, filters, a=0.01, dr=0.05, depth=3):
    x1 = Conv_block(x, filters, a, dr)

    xn = Conv_block(x1, filters, a, dr)
    cn = Add(axis=-1)([x1, xn])

    for _ in range(depth - 1):
        xn = Conv_block(cn, filters, a, dr)
        cn = Add(axis=-1)([x1, xn])

    return cn


def R2_block(x, filters, a=0.01, dr=0.05, depth=3):
    x1 = Conv_block(x, filters, a, dr)

    xn = Conv_block(x1, filters, a, dr)
    cn = Concatenate(axis=-1)([x1, xn])

    for _ in range(depth - 1):
        xn = Conv_block(cn, filters, a, dr)
        cn = Concatenate(axis=-1)([x1, xn])

    return cn
