import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers

class ReflectionPad2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1,1)):
        self.padding = tuple(padding)
        super(ReflectionPad2D, self).__init__()

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [[0,0],[padding_height,padding_height],[padding_width,padding_width],[0,0]]

        return tf.pad(input_tensor,padding_tensor,mode='REFLECT')

class ResNetBlock(layers.Layer):
    def __init__(self, num_filters, kernel_init=None, use_1x1conv=False, strides=1):
        super(ResNetBlock, self).__init__()
        
        if kernel_init == None:
            self.kernel_init = tf.keras.initializers.RandomNormal(0.0,0.02) # Used in the original implementation
        else:
            self.kernel_init = kernel_init 
        
        self.conv_1 = layers.Conv2D(num_filters, kernel_size=(3,3), strides=1, padding='valid', kernel_initializer = kernel_init, use_bias=False)
        self.conv_2 = layers.Conv2D(num_filters, kernel_size=(3,3), strides=1, padding='valid', kernel_initializer = kernel_init, use_bias=False)
        self.conv_3 = None

        if use_1x1conv == True:
            self.conv_3 = layers.Conv2D(num_filters, kernel_size=(1,1), strides=1)
        
        # Normalization layers
        self.instance_norm_1 = InstanceNormalization(axis=-1)
        self.instance_norm_2 = InstanceNormalization(axis=-1)

        # Reflection padding layers
        self.reflect_pad1 = ReflectionPad2D()
        self.reflect_pad2 = ReflectionPad2D()

    def call(self, X):
        # Reflection pad -> Conv -> Instance Norm -> Relu -> Reflection pad -> conv -> Instance Norm -> concat output and input
        
        Y = self.reflect_pad1(X)
        Y =  tf.keras.activations.relu(self.instance_norm_1(self.conv_1(X)))
        Y = self.reflect_pad2(X)
        Y = self.instance_norm_2(self.conv_2(Y))

        Y = tf.add(Y,X)

        return Y

      
