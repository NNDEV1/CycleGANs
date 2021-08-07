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

from utils import *

##Layer Classes##

#Conv2D layer
class ConvLayer():
    def __init__(self, num_filters=32, kernel_size=4, stride=2, padding="SAME", normalizer=None, activation=None, weights_init=tf.truncated_normal_initializer):

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.normalizer = normalizer
        self.activation = activation
        self.weights_init = weights_init

    def __call__(self, x):

        layer = slim.conv2d(x,self.num_filters, [self.kernel_size,self.kernel_size], [self.stride,self.stride],
                    padding=self.padding ,weights_initializer=self.weights_init, activation_fn=self.activation)
        
        return layer



#Conv2D Transposed Layer
class ConvTransposeLayer():
    def __init__(self, num_outputs=32, kernel_size=3, stride=2, 
                 padding="SAME", normalizer=None, activation=None):
        
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.normalizer = normalizer
        self.activation = activation

    def __call__(self, x):

        layer =  slim.conv2d_transpose(x, self.num_outputs, self.kernel_size, self.stride, self.padding, 
                                     normalizer_fn=self.normalizer, activation_fn=self.activation)
        
        return layer


#Padding Layer
class PaddingLayer():
    def __init__(self, padding_size, padding_type):

        self.padding_size = padding_size
        self.padding_type = padding_type

    def __call__(self, x):

        p = int(self.padding_size)

        layer = tf.pad(x,[[0, 0], [p, p], [p, p], [0, 0]],self.padding_type)

        return layer
      
      
#ResBlock for U_net Generator        
class ResBlock():
    def __init__(self, num_filters=32, kernel_size=4, stride=2, padding_size=1, padding_type="REFLECT", 
                 normalizer=None, activation=None, weights_init=tf.truncated_normal_initializer):

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_size = padding_size
        self.padding_type = padding_type
        self.normalizer = normalizer
        self.activation = activation
        self.weights_init = weights_init

    def __call__(self, x):
        layers = [
        PaddingLayer(self.padding_size, self.padding_type),
        ConvLayer(num_filters=self.num_filters, kernel_size=self.kernel_size, 
                  stride=self.stride, padding="VALID", weights_init=self.weights_init, 
                  normalizer=self.normalizer, activation=self.activation),
        PaddingLayer(self.padding_size, self.padding_type),
        ConvLayer(num_filters=self.num_filters, kernel_size=self.kernel_size, 
                  stride=self.stride, padding="VALID", weights_init=self.weights_init, 
                  normalizer=self.normalizer, activation=self.activation),                
                  ]

        res = forward(layers)(x) + x

        return res
