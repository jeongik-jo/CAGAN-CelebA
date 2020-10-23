import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as kr
import numpy as np
import os

inception_model = kr.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)

image_size = [64, 64]
channel_size = 3

batch_size = 32
latent_vector_dim = 512
latent_distribution_function = tf.random.normal

#attributes = ['Black_Or_Brown_Hair', 'Male', 'Smiling']
attributes = ['Young', 'Bangs', 'Black_Hair']
attributes_size = len(attributes)


save_image_size = 8
save_image_condition_vectors = tf.constant([
       [0, 0, 0],
       [0, 0, 1],
       [0, 1, 0],
       [0, 1, 1],
       [1, 0, 0],
       [1, 0, 1],
       [1, 1, 0],
       [1, 1, 1]],
    dtype='float32')
save_latent_vectors = latent_distribution_function([save_image_size, latent_vector_dim])

load_model = False
if load_model:
    save_latent_vectors = np.load('./models/latent_vectors.npy')
else:
    if not os.path.exists('./models'):
        os.makedirs('./models')
    np.save('./models/latent_vectors.npy', save_latent_vectors)

discriminator_optimizer = tf.optimizers.Adam(learning_rate=0.00001)
generator_optimizer = tf.optimizers.Adam(learning_rate=0.00001)

train_data_size = -1 # if -1, use all data
test_data_size = -1
is_acgan = False # if false, it's CAGAN
mixed_batch_training = False
ratio_per_epoch = 0.025
use_fcls = False

epochs = 50

adversarial_loss_weight = 1.0
classification_loss_weight = 10.0
is_wgan = False # if false, it's LSGAN
gp_loss_weight = 0.1
#DiscriminatorNormLayer = tfa.layers.InstanceNormalization
DiscriminatorNormLayer = kr.layers.BatchNormalization
GeneratorNormLayer = kr.layers.BatchNormalization

evaluate_model = True
epoch_per_evaluation = 1
test_dataset = 'test'
