import numpy as np
import tensorflow as tf

from efficientnet.keras import EfficientNetB4
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, MaxPooling2D

from blocks import residual_block

np.random.seed = 42

print("Initializing")


def efficientUnet():

    input_shape = (512, 512, 1)  # ------------------------<
    neurons = 8
    d = 0.1  # dropout

    encoder = EfficientNetB4(include_top=False, weights="imagenet", input_shape=input_shape)
    inputs = encoder.input

    c4 = encoder.layers[30].output
    # c4 = LeakyReLU(alpha=0.1)(c4)
    c3 = encoder.layers[92].output
    # c3 = LeakyReLU(alpha=0.1)(c3)
    c2 = encoder.layers[154].output
    # c2 = LeakyReLU(alpha=0.1)(c2)

    c1 = encoder.layers[342].output
    c1 = LeakyReLU(alpha=0.1)(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(d)(p1)

    c0 = residual_block(p1, neurons * 32)
    c0 = residual_block(c0, neurons * 32)

    # decoder

    u1 = Conv2DTranspose(neurons * 16, (2, 2), strides=(2, 2), padding="same")(c0)
    u1 = concatenate([u1, c1])
    u1 = Dropout(d)(u1)
    u1 = residual_block(u1, neurons * 16)

    u2 = Conv2DTranspose(neurons * 8, (2, 2), strides=(2, 2), padding="same")(u1)
    u2 = concatenate([u2, c2])
    u2 = Dropout(d)(u2)
    u2 = residual_block(u2, neurons * 8)

    u3 = Conv2DTranspose(neurons * 4, (2, 2), strides=(2, 2), padding="same")(u2)
    u3 = concatenate([u3, c3])
    u3 = Dropout(d)(u3)
    u3 = residual_block(u3, neurons * 4)

    u4 = Conv2DTranspose(neurons * 2, (2, 2), strides=(2, 2), padding="same")(u3)
    u4 = concatenate([u4, c4])
    u4 = Dropout(d)(u4)
    u4 = residual_block(u4, neurons * 2)

    out = Conv2D(1, (1, 1), activation="sigmoid")(u4)

    model = tf.keras.Model(inputs=[inputs], outputs=[output])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
