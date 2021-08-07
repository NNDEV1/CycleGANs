import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers

from layers import *

def build_discriminator(input_shape, k_init):

    inp = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, (4, 4), kernel_initializer=k_init, strides=2, padding='same')(inp)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(128, (4, 4), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(256, (4, 4), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(256, (4, 4), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Conv2D(512, (4, 4), kernel_initializer=k_init, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    output_x = tf.keras.layers.Conv2D(1, kernel_size=(4, 4), padding='same', kernel_initializer=k_init)(x)
    output_x = InstanceNormalization(axis=-1)(output_x)
    output_x = tf.keras.layers.LeakyReLU(alpha=0.2)(output_x)

    return tf.keras.Model(inputs=inp, outputs=output_x)


