import tensorflow as tf
import tensorflow.keras as kr
import tensorflow.keras.preprocessing.image as image
import Layers
import os
import HyperParameters as HP
import numpy as np


class Generator(object):
    def __init__(self):
        latent_vector = kr.Input([HP.latent_vector_dim])

        if HP.is_acgan:
            condition_vector = kr.Input([HP.attributes_size])
        else:
            condition_vector = kr.Input([HP.attributes_size * 2])

        input_vector = tf.concat([condition_vector, latent_vector], axis=-1)

        model_output = kr.layers.Dense(units=4 * 4 * 1024)(input_vector)
        model_output = kr.layers.Reshape(target_shape=[4, 4, 1024])(model_output)

        for _ in range(4):
            model_output = Layers.Upscale(conv_depth=0)(model_output)

        model_output = Layers.ToRGB()(model_output)
        self.model = kr.Model([condition_vector, latent_vector], model_output)

    def save_images(self, epoch):
        if not os.path.exists('./results/images'):
            os.makedirs('./results/images')

        images = []
        for i in range(HP.save_image_size):
            condition_vectors = HP.save_image_condition_vectors
            latent_vectors = tf.tile(tf.expand_dims(HP.save_latent_vectors[i], axis=0), [condition_vectors.shape[0], 1])

            if not HP.is_acgan:
                condition_vectors = tf.concat([condition_vectors, 1 - condition_vectors], axis=-1)

            fake_images = self.model([condition_vectors, latent_vectors])
            images.append(np.hstack(fake_images))
        image.save_img(path='./results/images/fake %d.png' % epoch, x=np.vstack(images))

    def save(self):
        if not os.path.exists('./models'):
            os.makedirs('models')

        self.model.save_weights('./models/generator.h5')

    def load(self):
        self.model.load_weights('./models/generator.h5')


class Discriminator(object):
    def __init__(self):
        model_output = input_image = kr.Input(shape=HP.image_size + [HP.channel_size])
        model_output = kr.layers.Conv2D(filters=64, kernel_size=5,
                                        padding='same', activation=tf.nn.leaky_relu, use_bias=False)(model_output)
        model_output = HP.DiscriminatorNormLayer()(model_output)

        for _ in range(4):
            model_output = Layers.DownScale(conv_depth=0)(model_output)

        model_output = kr.layers.Flatten()(model_output)
        output_adversarial_value = kr.layers.Dense(units=1, activation='linear', dtype='float32')(model_output)

        if HP.is_acgan:
            output_classification_value = kr.layers.Dense(units=HP.attributes_size, activation='linear', dtype='float32')(model_output)
        else:
            output_classification_value = kr.layers.Dense(units=HP.attributes_size * 2, activation='linear', dtype='float32')(model_output)

        self.model = kr.Model(input_image, [output_adversarial_value, output_classification_value])

    def save(self):
        if not os.path.exists('./models'):
            os.makedirs('models')

        self.model.save_weights('./models/discriminator.h5')

    def load(self):
        self.model.load_weights('./models/discriminator.h5')
