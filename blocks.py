from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, LeakyReLU


def residual_block(x, filters):
    y = Conv2D(filters, (3, 3), strides=(1, 1), padding="same", activation=None)(x)
    y = LeakyReLU(alpha=0.1)(y)
    y = BatchNormalization()(y)

    y = Conv2D(filters, (3, 3), strides=(1, 1), padding="same", activation=None)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = BatchNormalization()(y)

    out = Add()([x, y])
    return out
