from keras.layers import (Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Flatten, Input, LeakyReLU, Reshape)
from keras.models import Model

LATENT = 100


def build_generator():
    z = Input(shape=(LATENT,))
    x = Dense(8 * 8 * 256)(z)
    x = Reshape((8, 8, 256))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for filters in (128, 64):
        x = Conv2DTranspose(filters, 5, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = Conv2DTranspose(3, 5, strides=2, padding='same', activation='sigmoid')(x)
    return Model(z, x)


def build_discriminator():
    img = Input(shape=(64, 64, 3))
    x = img
    for filters in (64, 128, 256):
        x = Conv2D(filters, 5, strides=2, padding='same')(x)
        x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(img, x)


def build_gan(gen, disc):
    disc.trainable = False
    z = Input(shape=(LATENT,))
    return Model(z, disc(gen(z)))
