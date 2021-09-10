from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Dropout, LeakyReLU


def conv_block(x, filters: int, a=0.01, dr=0.05, k=3, s=1):
    """
    Custom convolution block
    conv > batch_norm > leakyReLU > dropout

    Args:
        x: Input tensor.
        filters (int): No. of filters in convolution layer.
        a (float, optional): Leakage rate for ReLU. Defaults to 0.01.
        dr (float, optional): Dropout rate. Defaults to 0.05.
        k (int, optional): Kernel size. Defaults to 3.
        s (int, optional): Stride. Defaults to 1.

    Returns:
        Output tensor
    """
    x = Conv2D(filters, (k, k), strides=(s, s), padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=a)(x)
    x = Dropout(rate=dr)(x)

    return x


def residual_block(x, filters: int, a=0.01, dr=0.05, depth=2):
    """
    Residual block with skip connection

    Args:
        x: Input tensor.
        filters (int): No. of filters in convolution layer.
        a (float, optional): Leakage rate for ReLU. Defaults to 0.01.
        dr (float, optional): Dropout rate. Defaults to 0.05.
        depth (int, optional): Depth of skip connection. Defaults to 2.

    Returns:
        Output tensor
    """
    y = conv_block(x, filters, a, dr)
    for _ in range(depth - 1):
        y = conv_block(y, filters, a, dr)
    y = Add()([x, y])

    return y


def recurrent_block(x, filters: int, a=0.01, dr=0.05, depth=3):
    """
    Recurrent block with feedback connection

    Args:
        x: Input tensor.
        filters (int): No. of filters in convolution layer.
        a (float, optional): Leakage rate for ReLU. Defaults to 0.01.
        dr (float, optional): Dropout rate. Defaults to 0.05.
        depth (int, optional): Number of recurrent feedbacks. Defaults to 3.

    Returns:
        Output tensor
    """
    y = conv_block(x, filters, a, dr)
    y = Add()([x, y])
    for _ in range(depth - 1):
        y = conv_block(y, filters, a, dr)
        y = Add()([x, y])

    return y


def r2_block(x, filters: int, a=0.01, dr=0.05, rec_depth=3, res_depth=2):
    """
    Recurrent residual block

    Args:
        x: Input tensor.
        filters (int): No. of filters in convolution layer.
        a (float, optional): Leakage rate for ReLU. Defaults to 0.01.
        dr (float, optional): Dropout rate. Defaults to 0.05.
        rec_depth (int, optional): Number of recurrent feedbacks. Defaults to 3.
        res_depth (int, optional): Depth of skip connection. Defaults to 2.

    Returns:
        Output tensor
    """
    y = recurrent_block(x, filters, a, dr, rec_depth)
    for _ in range(res_depth - 1):
        y = recurrent_block(y, filters, a, dr, rec_depth)
    y = Concatenate(axis=-1)([x, y])

    return y


def dense_block(x, filters: int, a=0.01, dr=0.05, depth=4):
    """
    DenseNet block with skip connections

    Args:
        x: Input tensor.
        filters (int): No. of filters in convolution layer.
        a (float, optional): Leakage rate for ReLU. Defaults to 0.01.
        dr (float, optional): Dropout rate. Defaults to 0.05.
        depth (int, optional): Number of layers. Defaults to 4.

    Returns:
        Output tensor
    """
    for _ in range(depth):
        xn = conv_block(x, filters, a, dr)
        x = Concatenate(axis=-1)([x, xn])

    return x


def fractal_block(x, filters: int, order: int, a=0.01, dr=0.05, join=True):
    """
    Generates a fractal connection block

    Args:
        x: Input tensor.
        filters (int): No. of filters in convolution layer.
        order (int): No. of fractal expansions.
        a (float, optional): Leakage rate for ReLU. Defaults to 0.01.
        dr (float, optional): Dropout rate. Defaults to 0.05.
        join (bool, optional): Joins layer outputs. Defaults to True.

    Returns:
        Output tensor
    """

    def flatten(box: list) -> list:
        """
        Deep flattens list of tensors.
        """
        if len(box) == 1:
            result = flatten(box[0]) if type(box[0]) == list else box
        elif type(box[0]) == list:
            result = flatten(box[0]) + flatten(box[1:])
        else:
            result = [box[0]] + flatten(box[1:])
        return result

    right = conv_block(x, filters, a, dr)
    if order > 2:
        left_a = fractal_block(x, filters, order - 1, a, dr, True)
        left_b = fractal_block(left_a, filters, order - 1, a, dr, False)
    else:
        left_a = conv_block(x, filters, a, dr)
        left_b = conv_block(left_a, filters, a, dr)

    if join:
        return Concatenate(axis=-1)(flatten([left_b, right]))
    else:
        return [left_b, right]
