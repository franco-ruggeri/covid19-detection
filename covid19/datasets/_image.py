import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import random_rotation, random_shift, random_zoom, random_brightness

BATCH_SIZE = 32


def _augment(image):
    image = random_rotation(image, 10, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant')
    image = random_shift(image, 0.1, 0.1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant')
    image = random_brightness(image, (0.9, 1.1))
    image = random_zoom(image, (0.85, 1.15), row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant')
    image = tf.image.random_flip_left_right(image)
    return image


def image_dataset_from_directory(dataset_path, image_size, augmentation=False, shuffle=True):
    """
    Generates a tf.data.Dataset from images in a directory.

    :param dataset_path: path to the dataset. The directory must contain one sub-directory for each class.
    :param image_size: tuple (height, width), image size
    :param augmentation: bool, whether to augment the data (horizontal flip, rotation, shift, brightness, zoom)
    :param shuffle: bool, whether to shuffle
    :return: (dataset, info), where dataset is a tf.data.Dataset and info is a dictionary containing useful information
        (e.g. n_batches).
    """
    # flow from directory
    image_flow = ImageDataGenerator().flow_from_directory(dataset_path, target_size=image_size, batch_size=BATCH_SIZE,
                                                          shuffle=shuffle)
    n_classes = image_flow.num_classes
    n_batches = len(image_flow)

    # tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(lambda: image_flow, output_types=(tf.float32, tf.float32),
                                             output_shapes=((None, image_size[0], image_size[1], 3), (None, n_classes)))
    if augmentation:
        dataset = dataset.unbatch()
        dataset = dataset.map(lambda image, label: (tf.numpy_function(_augment, [image], tf.float32), label),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.take(n_batches)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # useful info
    info = {
        'n_classes': n_classes,
        'n_images': len(image_flow.classes),
        'n_batches': n_batches,
        'class_labels': image_flow.class_indices,
        'batch_size': BATCH_SIZE,
        'labels': image_flow.classes
    }

    return dataset, info
