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

from cyclegan import CycleGAN

from os import listdir
from matplotlib import image
from PIL import Image
from keras.preprocessing.image import array_to_img, img_to_array
# load all images in a directory

dataB = list()
for filename in listdir('/content/horse2zebra/trainB/'):
	img = Image.open('/content/horse2zebra/trainB/'+filename)
	img = img_to_array(img)
	if img.shape==(256, 256, 3):
    	
		img_data = image.imread('/content/horse2zebra/trainB/' + filename)
		# store loaded image
		dataB.append(img_data)
		print('> loaded %s %s' % (filename, img_data.shape))

dataA = list()
for filename in listdir('/content/horse2zebra/trainA/'):
	# load image
	img_data = image.imread('/content/horse2zebra/trainA/' + filename)
	# store loaded image
	dataA.append(img_data)
	print('> loaded %s %s' % (filename, img_data.shape))
  
dataA = np.asarray(dataA, np.float32)
dataB = np.asarray(dataB, np.float32)


img_dim = (256, 256, 3)
num_train = dataA.shape[0]
dataA = (dataA / 255.0 - 0.5)*2
dataB = (dataB / 255.0 - 0.5)*2

tf.keras.backend.clear_session()
hyperparams = {
        'img_dim': img_dim,
        'batch_size' : 1,
        'num_examples' : num_train,
        'num_epochs' : 150,
        'learning_rate' : 0.0002,
        'weight_stdev' : 0.02,
        'n_res_blocks' : 9,
        'pool_size' : 50,
        'normalizer' : instance_norm,
        'lambda_A' : 10.0,
        'lambda_B' : 10.0,
    }
LOG_DIR = "log"
METAGRAPH_DIR = "out"
PLOTS_DIR = "png"

model = CycleGAN(dataA,dataB,hyperparams,img_dim)

model.train()
