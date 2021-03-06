import tensorflow as tf
import numpy as np
import os
import sys
import numpy as np
from PIL import Image, ImageOps
import sys
import matplotlib.pyplot as plt

#Utilities

def get_shape(tensor):
    return tensor.get_shape().as_list()

def inst_norm(tensor): # Instance Normalization custom layer
    epsilon = 1e-5
    with tf.variable_scope('in'):
        scale = tf.get_variable('scale', initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02), shape=[get_shape(tensor)[-1]])
        center = tf.get_variable('center', initializer=tf.zeros_initializer(dtype=tf.float32), shape=[get_shape(tensor)[-1]])
        instance_mean, instance_var = tf.nn.moments(tensor, axes=[1, 2], keep_dims=True)

        return scale * ((tensor - instance_mean) / tf.sqrt(instance_var + epsilon)) + center

def lkrelu(x, slope=0.01):
    return tf.maximum(slope * x, x)
