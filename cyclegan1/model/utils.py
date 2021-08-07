##Imports##

import functools
from functools import partial
import tensorflow as tf
import numpy as np
import h5py
import scipy
from imageio import imwrite, imread
from skimage.transform import resize
from os import listdir
from os.path import isfile, join, isdir
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.misc as sp
import time
import IPython.display
import tqdm # making loops prettier
import ipywidgets as widgets
import math
from ipywidgets import interact, interactive, fixed
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as tflayers

##Utils##

def compose(func_1, func_2, unpack=False):

    if not callable(func_1):
        raise TypeError("First argument to compose must be callable")
    if not callable(func_2):
        raise TypeError("Second argument to compose must be callable")
    
    if unpack:
        def composition(*args, **kwargs):
            return func_1(*func_2(*args, **kwargs))
    else:
        def composition(*args, **kwargs):
            return func_1(func_2(*args, **kwargs))
    return composition

def compose_all(*args):

    return partial(functools.reduce, compose)(*args)

def forward(layers):

    return compose_all(reversed(layers))

def leakyrelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * x + f2 * abs(x) 

#Should use Standard Deviation
def instance_norm(x, epsilon=1e-7, name="instance_norm"):
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, axes=(1,2), keep_dims=True)
        
        return (x - mean)/tf.sqrt(var + epsilon)

