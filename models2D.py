import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np
from SensingModel2D import *

def sensing_model(latent_dim, trainable):
    model = tf.keras.Sequential([
      MaskLayer_latendim(latent_dim, trainable=trainable),
    ], name='mask_model')
    return model

def build_model(recons_net, sensing_model, input_size):
    inputs = Input(input_size)
    corrupted = sensing_model(inputs)
    outputs = recons_net(corrupted)
    return Model(inputs, outputs)

def build_model2(recons_net, input_size):
    inputs = Input(input_size)
    corrupted = sensing_model(inputs)
    outputs = recons_net(corrupted)
    return Model(inputs, outputs)