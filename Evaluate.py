import tensorflow.keras as kr
import tensorflow as tf
import tensorflow_probability as tfp
import HyperParameters as HP
from scipy.linalg import sqrtm
import numpy as np
import os
import matplotlib.pyplot as plt


def features_to_fid(real_features, fake_features):
    real_features_mean = tf.reduce_mean(real_features, axis=0)
    fake_features_mean = tf.reduce_mean(fake_features, axis=0)

    mean_difference = tf.reduce_sum((real_features_mean - fake_features_mean) ** 2)
    real_cov, fake_cov = tfp.stats.covariance(real_features), tfp.stats.covariance(fake_features)
    cov_mean = sqrtm(tf.matmul(real_cov, fake_cov))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    cov_difference = tf.linalg.trace(real_cov + fake_cov - 2.0 * cov_mean)

    fid = mean_difference + cov_difference

    return fid


@tf.function
def data_to_features(generator: kr.Model, data):
    #real_images = tf.image.resize(tf.cast(data['image'], dtype='float32') / 127.5 - 1, HP.image_size)
    real_images = (tf.cast(data['image'], dtype='float32') / 127.5 - 1)[:, 45:173, 25:153, :]
    real_images = tf.image.resize(real_images, HP.image_size)
    temp_dict = {}
    for key in data['attributes']:
        temp_dict[key] = data['attributes'][key]
    temp_dict['Black_Or_Brown_Hair'] = tf.logical_or(temp_dict['Black_Hair'], temp_dict['Brown_Hair'])
    real_labels = tf.cast(tf.stack([temp_dict[attribute] for attribute in HP.attributes], axis=-1), dtype='float32')

    batch_size = real_images.shape[0]
    latent_vectors = HP.latent_distribution_function([batch_size, HP.latent_vector_dim])
    fake_labels = tf.cast(tf.random.uniform([batch_size, HP.attributes_size], minval=0, maxval=2, dtype='int32'),
                          dtype='float32')

    if HP.is_acgan:
        condition_vectors = fake_labels
    else:
        condition_vectors = tf.concat([fake_labels, 1 - fake_labels], axis=-1)
    fake_images = generator([condition_vectors, latent_vectors], training=False)

    real_images = tf.image.resize(real_images, [299, 299])
    fake_images = tf.image.resize(fake_images, [299, 299])

    real_features = HP.inception_model(real_images)
    fake_features = HP.inception_model(fake_images)

    return (real_features, real_labels), (fake_features, fake_labels)


def get_average_fid(generator: kr.Model, test_dataset: tf.data.Dataset):
    test_dataset = test_dataset.shuffle(10000).batch(HP.batch_size).prefetch(1)

    separated_real_features = [[] for _ in range(2 ** HP.attributes_size)]
    separated_fake_features = [[] for _ in range(2 ** HP.attributes_size)]
    index_weights = tf.constant([2 ** n for n in range(HP.attributes_size)], dtype='float32')

    for test_data in test_dataset:
        (real_features, real_labels), (fake_features, fake_labels) = data_to_features(generator, test_data)
        real_indexes = tf.cast(tf.reduce_sum(real_labels * tf.expand_dims(index_weights, axis=0), axis=1), dtype='int32')
        fake_indexes = tf.cast(tf.reduce_sum(fake_labels * tf.expand_dims(index_weights, axis=0), axis=1), dtype='int32')

        for real_feature, real_index in zip(real_features, real_indexes):
            separated_real_features[real_index].append(real_feature)
        for fake_feature, fake_index in zip(fake_features, fake_indexes):
            separated_fake_features[fake_index].append(fake_feature)

    separated_real_features = [tf.convert_to_tensor(real_features) for real_features in separated_real_features]
    separated_fake_features = [tf.convert_to_tensor(fake_features) for fake_features in separated_fake_features]

    fids = []
    for real_features, fake_features in zip(separated_real_features, separated_fake_features):
        if real_features.shape[0] == 0 or fake_features.shape[0] == 0:
            continue
        fids.append(features_to_fid(real_features, fake_features))

    return tf.reduce_mean(fids)


def save_graph(fids):
    if not os.path.exists('./results'):
        os.makedirs('./results')
    epochs = [i + 1 for i in range(len(fids))]

    plt.plot(epochs, fids)
    plt.xlabel('epochs')
    plt.ylabel('average fid')

    plt.savefig('./results/fids.png')
    np.savetxt('./results/fids.txt', np.array(fids), fmt='%f')