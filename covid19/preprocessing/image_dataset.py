from pathlib import Path
import tensorflow as tf

BUFFER_SIZE = 16 * 1024     # 16 KB


def _load_image(filepath, image_size):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.cond(tf.shape(image)[2] == 1, lambda: tf.image.grayscale_to_rgb(image), lambda: image)
    image = tf.image.resize(image, image_size)
    return image


def image_dataset_from_directory(dataset_path, image_size, batch_size, shuffle=True):
    """
    Creates a tf.data.Dataset.

    Each samples is a tuple (image, label), where label is one-hot encoded.
    Gray-scale images are converted to RGB and the possible 4rd channel is dropped. All the images will be RGB.

    :param dataset_path: path to the dataset. The directory must contain one subdirectory for each class.
    :param image_size: tuple (height, width) which the images are resized to
    :param batch_size: integer, batch size
    :param shuffle: bool, whether to shuffle
    :return: tuple (balanced_dataset, class_indexes), where class_labels is a dictionary (name -> label)
    """
    dataset_path = Path(dataset_path)
    class_paths = sorted([c for c in dataset_path.iterdir()])
    n_classes = len(class_paths)
    class_indexes = {}

    def load_image(filepath):
        return _load_image(filepath, image_size)

    dataset = None
    for label, class_path in enumerate(class_paths):
        class_indexes[class_path.name] = label
        label = tf.one_hot(label, n_classes)    # one-hot encoding (for categorical cross entropy)
        filepath_ds = tf.data.Dataset.list_files(str(dataset_path / '*/*')).cache()
        if shuffle:
            filepath_ds = filepath_ds.shuffle(BUFFER_SIZE)
        image_ds = filepath_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensors(label).repeat()
        class_ds = (
            tf.data.Dataset.zip((image_ds, label_ds))
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        if dataset is None:
            dataset = class_ds
        else:
            dataset.concatenate(class_ds)
    return dataset, class_indexes


def image_balanced_dataset_from_directory(dataset_path, image_size, batch_size):
    """
    Creates a tf.data.Dataset generating (infinitely) balanced batches. The strategy used is resampling, i.e. the
    batches are filled by sampling uniformly across different datasets, one for each class.

    Each samples is a tuple (image, label), where label is one-hot encoded.
    Shuffling is enabled, since a balanced dataset is thought to be used for training, not evaluation.
    Gray-scale images are converted to RGB and the possible 4rd channel is dropped. All the images will be RGB.

    :param dataset_path: path to the dataset. The directory must contain one subdirectory for each class.
    :param image_size: tuple (height, width) which the images are resized to
    :param batch_size: integer, batch size
    :param shuffle: bool, whether to shuffle
    :return: tuple (balanced_dataset, class_indexes, n_images), where class_labels is a dictionary (name -> label)
    """
    dataset_path = Path(dataset_path)
    class_paths = sorted([c for c in dataset_path.iterdir()])
    n_classes = len(class_paths)
    n_images = 0
    class_indexes = {}
    datasets = []

    def load_image(filepath):
        return _load_image(filepath, image_size)

    # make one dataset per class
    for label, class_path in enumerate(class_paths):
        n_images += len(list(class_path.iterdir()))
        class_indexes[class_path.name] = label
        label = tf.one_hot(label, n_classes)    # one-hot encoding (for categorical cross entropy)
        image_ds = (
            tf.data.Dataset.list_files(str(class_path / '*'))
            .cache()                            # file paths fit in memory
            .shuffle(BUFFER_SIZE)               # file paths require little memory for shuffling
            .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .repeat()
        )
        label_ds = tf.data.Dataset.from_tensors(label).repeat()
        class_ds = tf.data.Dataset.zip((image_ds, label_ds))
        datasets.append(class_ds)

    # make balanced dataset
    balanced_ds = (
        tf.data.experimental.sample_from_datasets(datasets)     # sample uniformly across the datasets
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return balanced_ds, class_indexes, n_images
