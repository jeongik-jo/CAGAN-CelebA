import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as HP


class Upscale(kr.layers.Layer):
    def __init__(self, conv_depth):
        super().__init__()
        self.conv_depth = conv_depth

    def build(self, input_shape):
        self.upscaling_layer = kr.layers.Conv2DTranspose(filters=int(input_shape[-1] / 2), kernel_size=5, strides=2,
                                                         padding='same', use_bias=False)
        self.upscaling_norm_layer = HP.GeneratorNormLayer()
        self.conv_layers = [kr.layers.Conv2D(filters=int(input_shape[-1] / 2), kernel_size=5, padding='same',
                                             use_bias=False) for _ in range(self.conv_depth)]
        self.norm_layers = [HP.GeneratorNormLayer() for _ in range(self.conv_depth)]

    def call(self, inputs, **kwargs):
        conv_output = tf.nn.relu(self.upscaling_norm_layer(self.upscaling_layer(inputs)))
        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            conv_output = tf.nn.relu(norm_layer(conv_layer(conv_output)))

        return conv_output


class DownScale(kr.layers.Layer):
    def __init__(self, conv_depth):
        super().__init__()
        self.conv_depth = conv_depth

    def build(self, input_shape):
        self.downscaling_layer = kr.layers.Conv2D(filters=input_shape[-1] * 2, kernel_size=5, strides=2,
                                                   padding='same', use_bias=False)
        self.downscaling_norm_layer = HP.DiscriminatorNormLayer()

        self.conv_layers = [kr.layers.Conv2D(filters=input_shape[-1] * 2, kernel_size=5, padding='same',
                                             use_bias=False) for _ in range(self.conv_depth)]
        self.norm_layers = [HP.DiscriminatorNormLayer() for _ in range(self.conv_depth)]

    def call(self, inputs, **kwargs):
        conv_output = inputs
        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            conv_output = tf.nn.leaky_relu(norm_layer(conv_layer(conv_output)))
        conv_output = tf.nn.leaky_relu(self.downscaling_norm_layer(self.downscaling_layer(conv_output)))

        return conv_output


class ToRGB(kr.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.conv_layer = kr.layers.Conv2D(filters=3, kernel_size=1, padding='same', activation='tanh', dtype='float32')

    def call(self, inputs, **kwargs):
        return self.conv_layer(inputs)
