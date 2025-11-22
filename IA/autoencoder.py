import tensorflow as tf
from tensorflow.keras import layers, Model


def build_fourier_autoencoder(input_shape=(256, 256, 1)):

    # -------------------
    #     ENCODER
    # -------------------
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2,2), padding="same")(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2,2), padding="same")(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2,2), padding="same")(x)
    # encoded shape: (32, 32, 128)

    # -------------------
    #     DECODER
    # -------------------
    x = layers.Conv2DTranspose(128, (3,3), strides=2, activation="relu", padding="same")(encoded)
    x = layers.Conv2DTranspose(64, (3,3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3,3), strides=2, activation="relu", padding="same")(x)

    decoded = layers.Conv2D(1, (3,3), activation="linear", padding="same")(x)
    # reconstrução: mesmo shape da entrada

    autoencoder = Model(inputs, decoded)

    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    return autoencoder