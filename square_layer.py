from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class SquareMulLayer(Layer):
    def __init__(self, **kwargs):
        """Return a * squared(x), a is a learnable scalar."""
        super(SquareMulLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.coef = K.zeros(shape=(1,))
        self.trainable_weights = [self.coef]

    def call(self, x, mask=None):
        return self.coef * K.square(x)

    def get_output_shape_for(self, input_shape):
        return input_shape
