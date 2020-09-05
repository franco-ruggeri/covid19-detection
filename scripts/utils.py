import numpy as np
from covid19.models import ResNet50, COVIDNet
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

VERBOSE = 2
IMAGE_SIZE = (224, 224)


def get_model(architecture, weights, dataset_info, load_weights):
    n_classes = dataset_info['n_classes']
    if architecture == 'resnet50':
        model = ResNet50(n_classes, weights=weights)
    elif architecture == 'covidnet':
        model = COVIDNet(n_classes, weights=weights)
    else:
        raise ValueError('Invalid architecture')
    if load_weights is not None:
        model.load_weights(load_weights)
    return model


def get_callbacks(checkpoints_path, logs_path):
    filepath_checkpoint = checkpoints_path / 'epoch_{epoch:02d}'
    return [
        ModelCheckpoint(filepath=str(filepath_checkpoint), save_weights_only=True, verbose=VERBOSE),
        TensorBoard(log_dir=logs_path, profile_batch=0)
    ]


def get_class_weights(train_ds_info):
    total = train_ds_info['n_images']
    n_classes = train_ds_info['n_classes']
    class_weights = {}

    for class_name, class_label in train_ds_info['class_labels'].items():
        # scale weights by total / n_classes to keep the loss to a similar magnitude
        # see https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
        n = len(np.where(train_ds_info['labels'] == class_label)[0])
        class_weights[class_label] = (1 / n) * (total / n_classes)
    return class_weights
