import tensorflow as tf

from efficientnet.keras import EfficientNetB4
from tensorflow.keras.layers import BatchNormalization, concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU

from blocks import residual_block


def efficientB4ResUnet(input_shape):
    encoder = EfficientNetB4(include_top=False, weights="imagenet", input_shape=input_shape)

    # contracting level 4
    input = encoder.input
    en4 = Conv2D(24, (3, 3), strides=(1, 1), padding="same", activation=None)(input)
    en4 = LeakyReLU(alpha=0.1)(en4)
    en4 = BatchNormalization()(en4)

    # contracting level 3
    en3 = encoder.layers[25].output

    # contracting level 2
    en2 = encoder.layers[83].output

    # contracting level 1
    en1 = encoder.layers[141].output

    # middle level 0
    en0 = encoder.layers[317].output
    cv0 = residual_block(en0, 160)
    cv0 = residual_block(cv0, 160)
    up0 = Conv2DTranspose(56, (2, 2), strides=(2, 2), padding="same", name="upconvolution_1")(cv0)

    # expandsion level 1
    cv1 = concatenate([up0, en1])
    cv1 = residual_block(cv1, 112)
    up1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same", name="upconvolution_2")(cv1)

    # expandsion level 2
    cv2 = concatenate([up1, en2])
    cv2 = residual_block(cv2, 64)
    up2 = Conv2DTranspose(24, (2, 2), strides=(2, 2), padding="same", name="upconvolution_3")(cv2)

    # expandsion level 3
    cv3 = concatenate([up2, en3])
    cv3 = residual_block(cv3, 48)
    up3 = Conv2DTranspose(24, (2, 2), strides=(2, 2), padding="same", name="upconvolution_4")(cv3)

    # expandsion level 4
    cv4 = concatenate([up3, en4])
    cv4 = residual_block(cv4, 48)

    out = Conv2D(1, (1, 1), activation="sigmoid")(cv4)

    model = tf.keras.Model(inputs=[input], outputs=[out])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
