import tensorflow as tf

from blocks import Conv_block, Dense_block, R2_block, Residual_block
from efficientnet.keras import EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, LeakyReLU, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def efficientB7ResUnet(input_shape):
    a = 0.01
    dr = 0.05

    Encoder = EfficientNetB7(include_top=False, weights="imagenet", input_shape=input_shape)

    # contracting level 1
    input = Encoder.input
    en1 = Residual_block(input, 32, a, dr)
    en1 = Residual_block(en1, 32, a, dr)
    en1 = Residual_block(en1, 32, a, dr)

    en1 = Conv_block(en1, 32, a, dr)

    # contracting level 2
    en2 = Encoder.layers[49].output

    # contracting level 3
    en3 = Encoder.layers[152].output

    # contracting level 4
    en4 = Encoder.layers[255].output

    # middle level 5
    en5 = Encoder.layers[551].output
    cv5 = Residual_block(en5, 224, a, dr)
    cv5 = Residual_block(cv5, 224, a, dr)
    cv5 = Residual_block(cv5, 224, a, dr)
    up5 = Conv2DTranspose(224, (2, 2), strides=(2, 2), padding="same")(cv5)

    # expandsion level 4
    cv4 = Concatenate(axis=-1)([up5, en4])
    cv4 = Residual_block(cv4, 80, a, dr)
    cv4 = Residual_block(cv4, 80, a, dr)
    cv4 = Residual_block(cv4, 80, a, dr)
    up4 = Conv2DTranspose(80, (2, 2), strides=(2, 2), padding="same")(cv4)

    # expandsion level 3
    cv3 = Concatenate(axis=-1)([up4, en3])
    cv3 = Residual_block(cv3, 48, a, dr)
    cv3 = Residual_block(cv3, 48, a, dr)
    cv3 = Residual_block(cv3, 48, a, dr)
    up3 = Conv2DTranspose(48, (2, 2), strides=(2, 2), padding="same")(cv3)

    # expandsion level 2
    cv2 = Concatenate(axis=-1)([up3, en2])
    cv2 = Residual_block(cv2, 32, a, dr)
    cv2 = Residual_block(cv2, 32, a, dr)
    cv2 = Residual_block(cv2, 32, a, dr)
    up2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(cv2)

    # expandsion level 1
    cv1 = Concatenate(axis=-1)([up2, en1])
    cv1 = Residual_block(cv1, 32, a, dr)
    cv1 = Residual_block(cv1, 32, a, dr)
    cv1 = Residual_block(cv1, 32, a, dr)

    cv1 = Conv_block(cv1, 32, a, dr)

    out = Conv2D(1, (1, 1), activation="sigmoid")(cv1)

    model = tf.keras.Model(inputs=[input], outputs=[out])

    model.compile(optimizer=Adam(0.01), loss="binary_crossentropy", metrics=["acc", "mse"])

    return model


def efficientB7DenseUnet(input_shape):
    a = 0.01
    dr = 0.05
    depth = 4

    Encoder = EfficientNetB7(include_top=False, weights="imagenet", input_shape=input_shape)

    # contracting level 1
    input = Encoder.input
    en1 = Dense_block(input, 32, a, dr, depth)

    en1 = Conv_block(en1, 32, a, dr)

    # contracting level 2
    en2 = Encoder.layers[49].output

    # contracting level 3
    en3 = Encoder.layers[152].output

    # contracting level 4
    en4 = Encoder.layers[255].output

    # middle level 5
    en5 = Encoder.layers[551].output
    cv5 = Dense_block(en5, 224, a, dr, depth)
    up5 = Conv2DTranspose(224, (2, 2), strides=(2, 2), padding="same")(cv5)

    # expandsion level 4
    cv4 = Concatenate(axis=-1)([up5, en4])
    cv4 = Dense_block(cv4, 80, a, dr, depth)
    up4 = Conv2DTranspose(80, (2, 2), strides=(2, 2), padding="same")(cv4)

    # expandsion level 3
    cv3 = Concatenate(axis=-1)([up4, en3])
    cv3 = Dense_block(cv3, 48, a, dr, depth)
    up3 = Conv2DTranspose(48, (2, 2), strides=(2, 2), padding="same")(cv3)

    # expandsion level 2
    cv2 = Concatenate(axis=-1)([up3, en2])
    cv2 = Dense_block(cv2, 32, a, dr, depth)
    up2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(cv2)

    # expandsion level 1
    cv1 = Concatenate(axis=-1)([up2, en1])
    cv1 = Dense_block(cv1, 32, a, dr, depth)

    cv1 = Conv_block(cv1, 32, a, dr)

    out = Conv2D(1, (1, 1), activation="sigmoid")(cv1)

    model = tf.keras.Model(inputs=[input], outputs=[out])

    model.compile(optimizer=Adam(0.01), loss="binary_crossentropy", metrics=["acc", "mse"])

    return model


def efficientB7R2Unet(input_shape):
    a = 0.01
    dr = 0.1
    depth = 4

    Encoder = EfficientNetB7(include_top=False, weights="imagenet", input_shape=input_shape)

    # contracting level 1
    input = Encoder.input
    en1 = R2_block(input, 32, a, dr, depth)

    en1 = Conv_block(en1, 32, a, dr)

    # contracting level 2
    en2 = Encoder.layers[49].output

    # contracting level 3
    en3 = Encoder.layers[152].output

    # contracting level 4
    en4 = Encoder.layers[255].output

    # middle level 5
    en5 = Encoder.layers[551].output
    cv5 = R2_block(en5, 224, a, dr, depth)
    up5 = Conv2DTranspose(224, (2, 2), strides=(2, 2), padding="same")(cv5)

    # expandsion level 4
    cv4 = Concatenate(axis=-1)([up5, en4])
    cv4 = R2_block(cv4, 80, a, dr, depth)
    up4 = Conv2DTranspose(80, (2, 2), strides=(2, 2), padding="same")(cv4)

    # expandsion level 3
    cv3 = Concatenate(axis=-1)([up4, en3])
    cv3 = R2_block(cv3, 48, a, dr, depth)
    up3 = Conv2DTranspose(48, (2, 2), strides=(2, 2), padding="same")(cv3)

    # expandsion level 2
    cv2 = Concatenate(axis=-1)([up3, en2])
    cv2 = R2_block(cv2, 32, a, dr, depth)
    up2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(cv2)

    # expandsion level 1
    cv1 = Concatenate(axis=-1)([up2, en1])
    cv1 = R2_block(cv1, 32, a, dr, depth)

    cv1 = Conv_block(cv1, 32, a, dr)

    out = Conv2D(1, (1, 1), activation="sigmoid")(cv1)

    model = tf.keras.Model(inputs=[input], outputs=[out])

    model.compile(optimizer=Adam(0.01), loss="binary_crossentropy", metrics=["acc", "mse"])

    return model
