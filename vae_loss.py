# from keras.layers import Input, Dense, Lambda, Layer
# from keras.models import Model
from keras.layers import Layer
from keras import backend as K
from keras import metrics


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        self.original_dim = 784
        self.z_log_var = kwargs['z_log_var']
        self.z_mean = kwargs['z_mean']
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        # if vae_deconv then
        # x = K.flatten(x)
        # x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.original_dim * metrics.binary_crossentropy(
            x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean)
                                - K.exp(self.z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x
