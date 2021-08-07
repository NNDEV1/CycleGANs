import tensorflow as tf
import numpy as np
import os
import sys
import numpy as np
from PIL import Image, ImageOps
import sys
import matplotlib.pyplot as plt

from model_utils import *

class Generator(object):
    def __init__(self, name, inputs, ochan, stddev=0.02, center=True, scale=True, reuse=tf.AUTO_REUSE):
        self._stddev = stddev
        self._ochan = ochan
        with tf.variable_scope(name, initializer=tf.truncated_normal_initializer(stddev=self._stddev), reuse=tf.AUTO_REUSE):
            self._inputs = inputs
            self._resnet = self._build_resnet(self._inputs)


    def __getitem__(self, key):
        return self._resnet[key]

    def _build_conv_layer(self, name, inputs, k, rfsize, stride, 
                          use_in=True, f=tf.nn.relu, reflect=False):
        layer = dict()
        with tf.variable_scope(name):
            layer['filters'] = tf.get_variable('filters', [rfsize, rfsize, get_shape(inputs)[-1], k])

            if reflect:
                layer['conv'] = tf.nn.conv2d(tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'), 
                                             layer['filters'], strides=[1, stride, stride, 1], padding='VALID')
            else:
                layer['conv'] = tf.nn.conv2d(inputs, layer['filters'], strides=[1, stride, stride, 1], padding='SAME')

            layer['bn'] = inst_norm(layer['conv']) if use_in else layer['conv']
            layer['fmap'] = f(layer['bn'])

        return layer

    def _build_residual_layer(self, name, inputs, k, rfsize, blocksize=2, stride=1):
        layer = dict()
        with tf.variable_scope(name):
            with tf.variable_scope('layer1'):
                layer['filters1'] = tf.get_variable('filters1', [rfsize, rfsize, get_shape(inputs)[-1], k])
                layer['conv1'] = tf.nn.conv2d(tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'), 
                                             layer['filters1'], strides=[1, stride, stride, 1], padding="VALID")
                layer['bn1'] = inst_norm(layer['conv1'])
                layer['fmap1'] = tf.nn.relu(layer['bn1'])

            with tf.variable_scope('layer2'):
                layer['filters2'] = tf.get_variable('filters2', [rfsize, rfsize, get_shape(inputs)[-1], k])
                layer['conv2'] = tf.nn.conv2d(tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'), 
                                             layer['filters2'], strides=[1, stride, stride, 1], padding="VALID")
                layer['bn2'] = inst_norm(layer['conv2'])

            layer['fmap2'] = layer['bn2'] + inputs

        return layer
    
    def _build_deconv_layer(self, name, inputs, k, output_shape, rfsize):
        layer = dict()
        with tf.variable_scope(name):
            output_shape = [tf.shape(inputs)[0]] + output_shape
            layer['filters'] = tf.get_variable('filters', [rfsize, rfsize, output_shape[-1], get_shape(inputs)[-1]])
            layer['conv'] = tf.nn.conv2d_transpose(inputs, layer['filters'], output_shape=output_shape, 
                                                   strides=[1, 2, 2, 1], padding="SAME")
            layer['bn'] = inst_norm(tf.reshape(layer['conv'], output_shape))
            layer['fmap'] = tf.nn.relu(layer['bn'])
        
        return layer

    def _build_resnet(self, inputs):
        resnet = dict()

        inputs_shape = get_shape(inputs)
        width = inputs_shape[1]
        height = inputs_shape[2]

        with tf.variable_scope('resnet'):
            #Names of layer from paper
            resnet['l1'] = self._build_conv_layer('conv-1', inputs, k=32, rfsize=7, stride=1, reflect=True)
            resnet['l2'] = self._build_conv_layer('conv-2', resnet['l1']['fmap'], k=64, rfsize=3, stride=2)
            resnet['l3'] = self._build_conv_layer('conv-3', resnet['l2']['fmap'], k=128, rfsize=3, stride=2)
            resnet['l4'] = self._build_residual_layer('res-1', resnet['l3']['fmap'], k=128, rfsize=3, stride=1)
            resnet['l5'] = self._build_residual_layer('res-2', resnet['l4']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l6'] = self._build_residual_layer('res-3', resnet['l5']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l7'] = self._build_residual_layer('res-4', resnet['l6']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l8'] = self._build_residual_layer('res-5', resnet['l7']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l9'] = self._build_residual_layer('res-6', resnet['l8']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l10'] = self._build_residual_layer('res-7', resnet['l9']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l11'] = self._build_residual_layer('res-8', resnet['l10']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l12'] = self._build_residual_layer('res-9', resnet['l11']['fmap2'], k=128, rfsize=3, stride=1)
            resnet['l13'] = self._build_deconv_layer('deconv-1', resnet['l12']['fmap2'], k=64, output_shape=[width//2, height//2, 64], rfsize=3)
            resnet['l14'] = self._build_deconv_layer('deconv-2', resnet['l13']['fmap'], k=32, output_shape=[width, height, 32], rfsize=3)
            resnet['l15'] = self._build_conv_layer('conv-out', resnet['l14']['fmap'], f=tf.nn.tanh, k=get_shape(inputs)[-1], rfsize=7, stride=1, use_in=False, reflect=True)

        return resnet
