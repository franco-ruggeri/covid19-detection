import numpy as np
import matplotlib.pyplot as plt


def balanced_flow_from_directory(image_generator, batch_size, class_names, directory, shuffle, target_size):
    """Generator of balanced batches."""
    n_classes = len(class_names)
    batch_size_per_class = batch_size // n_classes
    if batch_size % n_classes != 0:
        batch_size_per_class += 1

    # one generator per class
    data_generators = [image_generator.flow_from_directory(batch_size=batch_size_per_class, classes=[k],
                                                           directory=directory, shuffle=shuffle,
                                                           target_size=target_size)
                       for k in class_names]

    batch_images = np.zeros((batch_size,) + data_generators[0].image_shape)
    batch_classes = np.zeros((batch_size, n_classes))
    rest = batch_size % n_classes
    n_filled = 0

    while True:
        # shuffle classes to randomize small imbalances due to batch_size non-perfectly divisible by n_classes
        classes = np.arange(n_classes)
        np.random.shuffle(classes)

        for i, k in enumerate(classes):
            images = next(data_generators[k])[0]

            n_to_add = batch_size // n_classes
            if i < rest:    # if batch_size is not perfectly divisible by n_classes, we have to fill it
                n_to_add += 1

            batch_images[n_filled:(n_filled+n_to_add)] = images[:n_to_add]
            batch_classes[n_filled:(n_filled+n_to_add), k] = 1
            n_filled += n_to_add
        yield batch_images, batch_classes

        # reset the necessary
        n_filled = 0
        batch_classes = np.zeros((batch_size, n_classes))
