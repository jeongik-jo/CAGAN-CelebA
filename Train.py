import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as HP


@tf.function
def _train(generator: kr.Model, discriminator: kr.Model, data: tf.Tensor, epoch):
    with tf.GradientTape(persistent=True) as tape:
        real_images = (tf.cast(data['image'], dtype='float32') / 127.5 - 1)[:, 45:173, 25:153, :]
        real_images = tf.image.resize(real_images, HP.image_size)
        temp_dict = {}
        for key in data['attributes']:
            temp_dict[key] = data['attributes'][key]
        temp_dict['Black_Or_Brown_Hair'] = tf.logical_or(temp_dict['Black_Hair'], temp_dict['Brown_Hair'])
        real_labels = tf.cast(tf.stack([temp_dict[attribute] for attribute in HP.attributes], axis=-1), dtype='float32')

        if not HP.is_acgan:
            real_labels = tf.concat([real_labels, 1 - real_labels], axis=-1)

        batch_size = real_images.shape[0]
        latent_vectors = HP.latent_distribution_function([batch_size, HP.latent_vector_dim])
        condition_vectors = tf.cast(tf.random.uniform([batch_size, HP.attributes_size], minval=0, maxval=2, dtype='int32'), dtype='float32')

        if not HP.is_acgan:
            condition_vectors = tf.concat([condition_vectors, 1 - condition_vectors], axis=-1)

        fake_images = generator([condition_vectors, latent_vectors], training=True)

        if HP.mixed_batch_training:
            slice_index = tf.cast(tf.minimum(HP.ratio_per_epoch * epoch, 0.5) * batch_size, dtype='int32')

            real_images0, real_images1 = real_images[:slice_index], real_images[slice_index:]
            fake_images0, fake_images1 = fake_images[:slice_index], fake_images[slice_index:]
            adversarial_values0, classification_values0 = discriminator(tf.concat([real_images0, fake_images1], axis=0),
                                                                        training=True)
            adversarial_values1, classification_values1 = discriminator(tf.concat([fake_images0, real_images1], axis=0),
                                                                        training=True)

            real_adversarial_values = tf.concat([adversarial_values0[:slice_index],
                                                 adversarial_values1[slice_index:]], axis=0)
            fake_adversarial_values = tf.concat([adversarial_values1[:slice_index],
                                                 adversarial_values0[slice_index:]], axis=0)

            real_classification_values = tf.concat([classification_values0[:slice_index],
                                                    classification_values1[slice_index:]], axis=0)
            fake_classification_values = tf.concat([classification_values1[:slice_index],
                                                    classification_values0[slice_index:]], axis=0)
        else:
            real_adversarial_values, real_classification_values = discriminator(real_images, training=True)
            fake_adversarial_values, fake_classification_values = discriminator(fake_images, training=True)

        if HP.is_acgan:
            real_classification_losses = tf.losses.binary_crossentropy(real_labels, tf.nn.sigmoid(real_classification_values))
            fake_classification_losses = tf.losses.binary_crossentropy(condition_vectors, tf.nn.sigmoid(fake_classification_values))

            if HP.is_wgan:
                discriminator_adversarial_losses = tf.squeeze(-real_adversarial_values + fake_adversarial_values)
                generator_adversarial_losses = tf.squeeze(-fake_adversarial_values)
            else:
                discriminator_adversarial_losses = tf.squeeze(tf.square(real_adversarial_values - 1)
                                                              + tf.square(fake_adversarial_values))
                generator_adversarial_losses = tf.squeeze(tf.square(fake_adversarial_values - 1))

            discriminator_losses = HP.adversarial_loss_weight * discriminator_adversarial_losses \
                                   + HP.classification_loss_weight * real_classification_losses
            if HP.use_fcls:
                discriminator_losses += HP.classification_loss_weight * fake_classification_losses

            generator_losses = HP.adversarial_loss_weight * generator_adversarial_losses \
                               + HP.classification_loss_weight * fake_classification_losses

        else:
            if HP.is_wgan:
                discriminator_losses = tf.reduce_sum(-real_classification_values * real_labels
                                                     + fake_classification_values * condition_vectors, axis=1)
                generator_losses = tf.reduce_sum(-fake_classification_values * condition_vectors, axis=1)
            else:
                discriminator_losses = tf.reduce_sum(tf.square(real_classification_values - 1) * real_labels
                                                     + tf.square(fake_classification_values) * condition_vectors, axis=1)
                generator_losses = tf.reduce_sum(tf.square(fake_classification_values - 1) * condition_vectors, axis=1)

        if HP.is_wgan:
            inner_vector = tf.random.uniform([batch_size, 1, 1, 1])
            inner_images = real_images * inner_vector + fake_images * (1 - inner_vector)

            with tf.GradientTape() as inner_tape:
                inner_tape.watch(inner_images)
                inner_adversarial_values, inner_classification_values = discriminator(inner_images, training=True)

                if HP.is_acgan:
                    score = tf.squeeze(inner_adversarial_values)
                else:
                    score = tf.reduce_mean(inner_classification_values, axis=1)

            gradients = inner_tape.gradient(score, inner_images)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
            gradient_penalty = tf.square(slopes - 1)

            discriminator_losses += HP.gp_loss_weight * gradient_penalty

    generator_gradients = tape.gradient(generator_losses, generator.trainable_variables)
    discriminator_gradients = tape.gradient(discriminator_losses, discriminator.trainable_variables)

    HP.generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    HP.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    del tape

    return tf.reduce_mean(discriminator_losses), tf.reduce_mean(generator_losses)


def train(generator: kr.Model, discriminator: kr.Model, train_dataset: tf.data.Dataset, epoch):
    train_dataset = train_dataset.shuffle(10000).batch(HP.batch_size).prefetch(1)
    discriminator_losses, generator_losses = [], []

    for data in train_dataset:
        discriminator_loss, generator_loss = _train(generator, discriminator, data, epoch)
        discriminator_losses.append(discriminator_loss)
        generator_losses.append(generator_loss)

    return tf.reduce_mean(discriminator_losses), tf.reduce_mean(generator_losses)
