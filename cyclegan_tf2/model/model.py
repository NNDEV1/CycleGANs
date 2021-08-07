import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers

from discriminator import build_discriminator
from generator import build_gen
from layers import *

def discriminator_loss(real, gen):

    loss_object_mse = tf.keras.losses.MeanSquaredError()
    real_loss = loss_object_mse(tf.ones_like(real), real)
    generated_loss = loss_object_mse(tf.zeros_like(gen), gen)
    total_discrim_loss = (real_loss + generated_loss) * 0.5
    
    return total_discrim_loss

def generator_loss(gen_output):
    
    loss_object_mse = tf.keras.losses.MeanSquaredError()
    gen_loss = loss_object_mse(tf.ones_like(gen_output), gen_output)

    return gen_loss

class CycleGAN(tf.keras.Model):
    def __init__(self, discrim_x, discrim_y, gen_G, gen_F, lambda_val_cycle=10, lambda_val_identity=0.5):
        super(CycleGAN, self).__init__()
        self.gen_G = gen_G
        self.gen_F = gen_F
        self.discrim_x = discrim_x
        self.discrim_y = discrim_y
        self.lambda_val_cycle = lambda_val_cycle
        self.lambda_val_identity = lambda_val_identity 
        
    def compile(self, discrim_x_optimizer, discrim_y_optimizer, gen_g_optimizer, gen_f_optimizer, gen_loss_fn, discrim_loss_fn):
        super(CycleGAN, self).compile()
        
        self.discrim_x_optimizer = discrim_x_optimizer
        self.discrim_y_optimizer = discrim_y_optimizer
        self.gen_G_optimizer = gen_g_optimizer
        self.gen_F_optimizer = gen_f_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.discrim_loss_fn = discrim_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()
        
    def train_step(self, data):
        real_x, real_y = data
        
        with tf.GradientTape(persistent=True) as tape:
            
            generated_y = self.gen_G(real_x, training=True)
            generated_x = self.gen_F(real_y,training=True)
            
            identity_y = self.gen_G(real_y,training=True)
            identity_x = self.gen_F(real_x,training=True)
            
            cycle_y = self.gen_G(generated_x,training=True)
            cycle_x = self.gen_F(generated_y,training=True)
            
            discrim_generated_x = self.discrim_x(generated_x,training=True)
            discrim_generated_y = self.discrim_y(generated_y,training=True)
            
            discrim_real_x = self.discrim_x(real_x,training=True)
            discrim_real_y = self.discrim_y(real_y,training=True)
            

            gen_G_loss = self.gen_loss_fn(discrim_generated_y)
            gen_F_loss = self.gen_loss_fn(discrim_generated_x)
            
            identity_loss_G = self.identity_loss_fn(real_y, identity_y) * self.lambda_val_cycle * self.lambda_val_identity
            identity_loss_F = self.identity_loss_fn(real_x, identity_x) * self.lambda_val_cycle * self.lambda_val_identity
            
            cycle_loss_G = self.cycle_loss_fn(real_y, generated_y) * self.lambda_val_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, generated_x) * self.lambda_val_cycle
            
            gen_G_total_loss = gen_G_loss + identity_loss_G + cycle_loss_G
            gen_F_total_loss = gen_F_loss + identity_loss_F + cycle_loss_F
                        
            d_loss_x = self.discrim_loss_fn(discrim_real_x, discrim_generated_x)
            d_loss_y = self.discrim_loss_fn(discrim_real_y, discrim_generated_y)
            
            
        gen_G_grads = tape.gradient(gen_G_total_loss, self.gen_G.trainable_variables)
        gen_F_grads = tape.gradient(gen_F_total_loss, self.gen_F.trainable_variables)


        discrim_x_grads = tape.gradient(d_loss_x, self.discrim_x.trainable_variables)
        discrim_y_grads = tape.gradient(d_loss_y, self.discrim_y.trainable_variables)


        self.gen_G_optimizer.apply_gradients(zip(gen_G_grads, self.gen_G.trainable_variables))
        self.gen_F_optimizer.apply_gradients(zip(gen_F_grads, self.gen_F.trainable_variables))


        self.discrim_x_optimizer.apply_gradients(zip(discrim_x_grads, self.discrim_x.trainable_variables))
        self.discrim_y_optimizer.apply_gradients(zip(discrim_y_grads, self.discrim_y.trainable_variables))
 
        return {
            'gen_G_loss': gen_G_total_loss,
            'gen_F_loss': gen_F_total_loss,
            'discrim_x_loss': d_loss_x,
            'discrim_y_loss': d_loss_y,}


