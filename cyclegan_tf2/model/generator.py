import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers

from layers import *

def build_gen(input_shape, k_init):
    inp = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (7, 7), kernel_initializer=k_init, strides=1, padding='same')(inp)
    x = InstanceNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)  

    x = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer=k_init, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)  

    x = ResNetBlock(128, kernel_init=k_init)(x)
    x = ResNetBlock(128, kernel_init=k_init)(x)
    x = ResNetBlock(128, kernel_init=k_init)(x)
    x = ResNetBlock(128, kernel_init=k_init)(x)
    x = ResNetBlock(128, kernel_init=k_init)(x)
    x = ResNetBlock(128, kernel_init=k_init)(x)
    x = ResNetBlock(128, kernel_init=k_init)(x)
    x = ResNetBlock(128, kernel_init=k_init)(x)
    x = ResNetBlock(128, kernel_init=k_init)(x)

    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), kernel_initializer=k_init, padding='same', strides=2)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), kernel_initializer=k_init, padding='same', strides=2)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2DTranspose(3, kernel_size=(7, 7), kernel_initializer=k_init, padding='same', strides=1)(x)
    output_layer = InstanceNormalization(axis=-1)(x)
    
    output_layer = tf.keras.layers.Activation('tanh')(output_layer)

    return tf.keras.Model(inputs=inp, outputs=output_layer)
