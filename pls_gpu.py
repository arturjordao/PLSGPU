"""Partial Least Squares GPU"""

# Author: Artur Jordao <arturjlcorreia@gmail.com>
#         Artur Jordao

import numpy as np
import keras

class PLSGPU():
    """Partial Least Squares GPU.

    Parameters
    ----------
    pls : sklear model

    batch_size : int or None, (default=None)
        The number of samples to be send for GPU.
    """

    def __init__(self, pls=None, batch_size=32):
        __name__ = 'Partial Least Squares GPU'
        self.batch_size = batch_size

        w = pls.x_rotations_
        x_mean = pls.x_mean_
        x_std = pls.x_std_
        d, c = w.shape #Original high dimensional feature space (R^d) and latent space (R^c)

        input = keras.layers.Input((d,))
        H = keras.layers.Lambda(lambda x: (x - x_mean) / x_std)(input)
        H = keras.layers.Dense(c, use_bias=False, trainable=False, name='pls')(H)

        self.pls_gpu = keras.models.Model([input], H)
        self.pls_gpu.get_layer(name='pls').set_weights([w])


    def transform(self, X):
        return self.pls_gpu.predict(X, batch_size=self.batch_size)
