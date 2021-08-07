import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import os

def decode_image(img):
    '''Decode jpg images and return 286,286,3 tensors'''

    img = tf.image.decode_jpeg(img, channels=3)

    return tf.image.resize(img, [286,286]) 

def preprocess_train_image(img):
    '''
    Applies to training images:
        - Left Right random flip
        - Random Crop to [256,256,3]
        - Normalize to pixel range of [-1,1] as done in the original implementation
    '''
    # Random flip
    img = tf.image.random_flip_left_right(img)

    # Random crop
    img = tf.image.random_crop(img, size=INPUT_SHAPE)

    # Normalize to [-1,1]
    img = tf.cast(img, dtype=tf.float32)
    return (img/127.5) - 1.0
  
  
def preprocess_test_image(img):
    '''
    Applies to test images
        - Resizes to [256,256,3]
        - Normalize to pixel range of [-1,1] as done in the original implementation
    '''

    # Resize
    img = tf.image.resize(img, INPUT_SHAPE[:-1])
    img = tf.cast(img, dtype=tf.float32)
    return (img/127.5) - 1.0
  
  
def load_train_image(filepath):
    '''
    Loads and preprocess training images
    '''

    img = tf.io.read_file(filepath)
    img = decode_image(img)
    img = preprocess_train_image(img)

    return img

def load_test_image(filepath):
    '''
    Loads and preprocess test images
    '''

    img = tf.io.read_file(filepath)
    img = decode_image(img)
    img = preprocess_test_image(img)

    return img
