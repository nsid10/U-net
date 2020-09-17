import numpy as np
import pandas as pd
import tensorflow as tf

from efficientnet import EfficientNetB4
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, MaxPooling2D

print("Initializing")


def efficientUnet():

    input_shape = 123456  # ------------------------<
    neurons = 8
    d = 0.1  # dropout

    encoder = EfficientNetB4(include_top=False, weights="imagenet", input_shape=input_shape)
    inputs = encoder.input

    conv1 = encoder.layers[342].output
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(neurons * 32, (3, 3), padding="same", activation="relu", name="middle")(pool1)
    conv2 = residual_block(conv2, neurons * 32)
    conv2 = LeakyReLU(alpha=0.1)(conv2)

    # decoder

    upcn3 = Conv2DTranspose(neurons * 16, (2, 2), strides=(2, 2), padding="same")(conv2)
    upcn3 = concatenate([upcn3, conv1])
    upcn3 = Dropout(d)(upcn3)
    conv3 = Conv2D(neurons * 16, (3, 3), padding="same", activation="relu")(upcn3)
    conv3 = residual_block(conv3, neurons * 16)
    conv3 = LeakyReLU(alpha=0.1)(conv3)

    upcn4 = Conv2DTranspose(neurons * 8, (2, 2), strides=(2, 2), padding="same")(conv3)
    en4 = encoder.layers[154].output
    upcn4 = concatenate([upcn4, en4])
    upcn4 = Dropout(d)(upcn4)
    conv4 = Conv2D(neurons * 8, (3, 3), padding="same", activation="relu")(upcn4)
    conv4 = residual_block(conv4, neurons * 8)
    conv4 = LeakyReLU(alpha=0.1)(conv4)

    upcn5 = Conv2DTranspose(neurons * 4, (2, 2), strides=(2, 2), padding="same")(conv4)
    en5 = encoder.layers[92].output
    upcn5 = concatenate([upcn5, en5])
    upcn5 = Dropout(d)(upcn5)
    conv5 = Conv2D(neurons * 4, (3, 3), padding="same", activation="relu")(upcn5)
    conv5 = residual_block(conv5, neurons * 4)
    conv5 = LeakyReLU(alpha=0.1)(conv5)

    upcn6 = Conv2DTranspose(neurons * 2, (2, 2), strides=(2, 2), padding="same")(conv5)
    en6 = encoder.layers[30].output
    upcn6 = concatenate([upcn6, en6])
    upcn6 = Dropout(d)(upcn6)
    conv6 = Conv2D(neurons * 2, (3, 3), padding="same", activation="relu")(upcn6)
    conv6 = residual_block(conv6, neurons * 2)
    conv6 = LeakyReLU(alpha=0.1)(conv6)

    out = Conv2D(1, (1, 1), activation="sigmoid")(conv6)

    model = tf.keras.Model(inputs=[inputs], outputs=[output])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model
