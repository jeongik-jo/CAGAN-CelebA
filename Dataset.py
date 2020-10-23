import HyperParameters as HP
import tensorflow_datasets as tfds


def load_dataset():
    celeb_a = tfds.image.CelebA().as_dataset(shuffle_files=True)

    if HP.train_data_size == -1:
        train_dataset = celeb_a['train']
    else:
        train_dataset = celeb_a['train'].take(HP.train_data_size)

    if HP.test_data_size == -1:
        test_dataset = celeb_a[HP.test_dataset]
    else:
        test_dataset = celeb_a[HP.test_dataset].take(HP.test_data_size)

    return train_dataset, test_dataset
