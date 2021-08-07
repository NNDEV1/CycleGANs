import tensorflow as tf
import numpy as np
import os
import sys
import numpy as np
from PIL import Image, ImageOps
import sys
import matplotlib.pyplot as plt

from train import *
from model.model_utils import *


A = np.load('/content/dataset_trainA.npy') / 255.
B = np.load('/content/dataset_trainB.npy') / 255.
tf.keras.backend.clear_session()
iters = 200 * min(A.shape[0], B.shape[0])
batch_size = 1

with tf.device('/gpu:0'):
    model = CycleGAN(256, 256, xchan=3, ychan=3)

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    start = 0

    sess.run(tf.global_variables_initializer())

    for step in range(start, iters):
        #  normalize because generator use tanh activation in its output layer
        a = 2. * np.expand_dims(A[np.random.randint(0, A.shape[0] - 1)], axis=0) - 1.
        b = 2. * np.expand_dims(B[np.random.randint(0, B.shape[0] - 1)], axis=0) - 1.

        d_a = 2. * np.expand_dims(A[np.random.randint(0, A.shape[0] - 1)], axis=0) - 1.
        d_b = 2. * np.expand_dims(B[np.random.randint(0, B.shape[0] - 1)], axis=0) - 1.

        (gxloss_curr, gyloss_curr), (dxloss_curr, dyloss_curr) = model.train_step(sess, a, b, d_a, d_b) #a: xs, b: ys
        print('Step %d: Gx loss: %f | Gy loss: %f | Dx loss: %f | Dy loss: %f' % (step, gxloss_curr, gyloss_curr, dxloss_curr, dyloss_curr))

        if step % 100 == 0:
            fig = plt.figure()
            fig.set_size_inches(15, 15)
            fig.subplots_adjust(left=0, bottom=0,
                                   right=1, top=1, wspace=0, hspace=0.1)

            for i in range(0, 12*12, 6):
                ra = np.random.randint(0, A.shape[0] - 1)
                rb = np.random.randint(0, B.shape[0] - 1)

                fig.add_subplot(12, 12, i + 1)
                plt.imshow(A[ra])
                plt.axis('off')

                fig.add_subplot(12, 12, i + 2)
                b_from_a = model.sample_gy(sess, 2. * np.expand_dims(A[ra], axis=0) - 1.)
                plt.imshow((b_from_a[0] + 1.) / 2.)
                plt.axis('off')

                fig.add_subplot(12, 12, i +3)
                plt.imshow(B[rb] )
                plt.axis('off')

                fig.add_subplot(12, 12, i + 4)
                a_from_b = model.sample_gx(sess, 2. * np.expand_dims(B[rb], axis=0) - 1.)
                plt.imshow((a_from_b[0] + 1.) / 2.)
                plt.axis('off')

                fig.add_subplot(12, 12, i + 5)
                identity_a = model.sample_gx(sess, b_from_a)
                plt.imshow((identity_a[0] + 1.) / 2.)
                plt.axis('off')

                fig.add_subplot(12, 12, i + 6)
                identity_b = model.sample_gy(sess, a_from_b)
                plt.imshow((identity_b[0] + 1.) / 2.)
                plt.axis('off')
            plt.savefig('/content/new_images/iter_%d.jpg' % step)
            plt.close()

        if step % 1000 == 0:
            # Save the model
            save_path = saver.save(sess, "/content/models/model.ckpt")
            print("Model saved in file: %s" % save_path)
            
            
            
