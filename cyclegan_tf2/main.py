import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import os

from model.model import CycleGAN
from image_utils import *

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, monitor_image_filepath, num_imgs=4):
        self.monitor_image_filepath = monitor_image_filepath
        self.num_imgs = num_imgs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            _, ax = plt.subplots(4, 2, figsize=(12, 12))
             
            for i, img in enumerate(train_photos.take(4)):

                prediction = self.model.gen_G(img)[0].numpy()
                prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
                img = (img[0] * 127.5 +127.5).numpy().astype(np.uint8)
                
                ax[i, 0].imshow(img)
                ax[i, 1].imshow(prediction)
                ax[i, 0].set_title("Input image")
                ax[i, 1].set_title("Translated image")
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")
                
                prediction = tf.keras.preprocessing.image.array_to_img(prediction)
                prediction.save(
                    "{monitor_image_filepath}/generated_img_{i}_{epoch}.png".format(monitor_image_filepath = self.monitor_image_filepath, i=i, epoch=epoch+1) 
                                )
            plt.show()
            plt.close()


PAINT_TRAIN_PATH = '/content/horse2zebra/trainB'
PHOTO_TRAIN_PATH = '/content/horse2zebra/trainA'
PAINT_TEST_PATH = '/content/horse2zebra/testB'
PHOTO_TEST_PATH = '/content/horse2zebra/testA'

MONITOR_IMAGE_FILEPATH = '/content/output'
CHECKPOINT_FILEPATH = '/content/ckpts'
FINAL_WEIGHTS_PATH = '/content/final_weights'

INPUT_SHAPE = [256, 256, 3]
DISCRIM_LR = 0.0002
DISCRIM_BETA = 0.5
GEN_LR = 0.0002
GEN_BETA = 0.5
EPOCHS = 100
K_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
SAVE_FORMAT = "tf"
AUTOTUNE = tf.data.experimental.AUTOTUNE


train_paintings = tf.data.Dataset.list_files(PAINT_TRAIN_PATH + '/*.jpg')
train_photos = tf.data.Dataset.list_files(PHOTO_TRAIN_PATH + '/*.jpg' )

test_paintings = tf.data.Dataset.list_files(PAINT_TEST_PATH + '/*.jpg')
test_photos = tf.data.Dataset.list_files(PHOTO_TEST_PATH + '/*.jpg')

train_paintings = train_paintings.map(load_train_image, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).batch(1)
train_photos = train_photos.map(load_train_image, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).batch(1)

test_paintings = test_paintings.map(load_test_image, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).batch(1)
test_photo = test_photos.map(load_test_image, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).batch(1)

generator_g = build_gen(input_shape=INPUT_SHAPE, k_init=K_INIT)
generator_f = build_gen(input_shape=INPUT_SHAPE, k_init=K_INIT)

discriminator_x = build_discriminator(input_shape=INPUT_SHAPE, k_init=K_INIT)
discriminator_y = build_discriminator(input_shape=INPUT_SHAPE, k_init=K_INIT)

c_gan_model = CycleGAN(discrim_x=discriminator_x, 
                       discrim_y=discriminator_y, 
                       gen_G=generator_g, 
                       gen_F=generator_f)

c_gan_model.compile(
    discrim_x_optimizer = tf.keras.optimizers.Adam(learning_rate=DISCRIM_LR, beta_1=DISCRIM_BETA),
    discrim_y_optimizer = tf.keras.optimizers.Adam(learning_rate=DISCRIM_LR, beta_1=DISCRIM_BETA),
    gen_g_optimizer = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=GEN_BETA),
    gen_f_optimizer = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=GEN_BETA),
    gen_loss_fn = generator_loss,
    discrim_loss_fn = discriminator_loss
)

monitor = GANMonitor(MONITOR_IMAGE_FILEPATH)
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_FILEPATH, save_weights_only=True, save_format=SAVE_FORMAT, save_freq=562*5
)

c_gan_model.fit(tf.data.Dataset.zip((train_photos, train_paintings)),
                epochs=EPOCHS,
                callbacks=[monitor, ckpt_callback],
                verbose=1,
                )

c_gan_model.save_weights(FINAL_WEIGHTS_PATH, save_format=SAVE_FORMAT)
