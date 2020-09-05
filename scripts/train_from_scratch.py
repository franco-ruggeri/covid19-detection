import argparse
import numpy as np
from pathlib import Path
from covid19.datasets import image_dataset_from_directory
from covid19.metrics import plot_learning_curves
from covid19.models import ResNet50, COVIDNet
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.metrics import CategoricalAccuracy, AUC, Precision, Recall
from tensorflow_addons.metrics import F1Score

VERBOSE = 2
IMAGE_SIZE = (224, 224)


def get_command_line_arguments():
    parser = argparse.ArgumentParser(description='Train COVID-19 detection model from scratch.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=str, help='path to the dataset')
    parser.add_argument('model', type=str,
                        help='path where to save trained model, learning curves, checkpoints and logs. Must be a '
                             'non-existing directory.')
    parser.add_argument('--architecture', type=str, default='resnet50',
                        help='architecture to use. Supported: resnet50, covidnet.')
    parser.add_argument('--class-weights', action='store_true', default=False,
                        help='compensate dataset imbalance using class weights')
    parser.add_argument('--data-augmentation', action='store_true', default=False, help='augment data during training')
    parser.add_argument('--load-weights', type=str, default=None,
                        help='path to weights to be loaded (useful for resuming training)')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epochs to skip (useful for resuming training)')
    parser.add_argument('--epochs', type=int, default=30, help='epochs of training')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--pretraining', action='store_true', default=False,
                        help='whether this run is for pretraining (the pretrained model will be saved)')
    return parser.parse_args()


def get_model(architecture, dataset_info, load_weights):
    n_classes = dataset_info['n_classes']
    if architecture == 'resnet50':
        model = ResNet50(n_classes, weights=None)
    elif architecture == 'covidnet':
        model = COVIDNet(n_classes, weights=None)
    else:
        raise ValueError('Invalid architecture')
    if load_weights is not None:
        model.load_weights(load_weights)
    return model


def get_loss():
    return CategoricalCrossentropy()


def get_metrics(dataset_info):
    # remarks:
    # - AUC and F1-score are computed with macro-average (we care a lot about the small COVID-19 class!)
    # - precision and recall are computed only on the COVID-19 class (again, it is the most important)
    n_classes = dataset_info['n_classes']
    metrics = [
        CategoricalAccuracy(name='accuracy'),
        AUC(name='auc', multi_label=True),      # multi_label=True => macro-average
        F1Score(name='f1-score', num_classes=n_classes, average='macro')
    ]
    if 'covid-19' in dataset_info['class_labels']:
        covid19_label = dataset_info['class_labels']['covid-19']
        metrics.append(Precision(name='precision_covid19', class_id=covid19_label))
        metrics.append(Recall(name='recall_covid19', class_id=covid19_label))
    return metrics


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


def main():
    # command-line arguments
    args = get_command_line_arguments()

    # prepare paths
    dataset_path = Path(args.data)
    models_path = Path(args.model)
    if models_path.is_dir():
        raise FileExistsError(str(models_path) + ' already exists')
    logs_path = models_path / 'logs'
    checkpoints_path = models_path / 'checkpoints'
    plots_path = models_path / 'training'
    models_path = models_path / 'models'
    models_path.mkdir(parents=True)
    checkpoints_path.mkdir()
    plots_path.mkdir()

    # build input pipeline
    train_ds, train_ds_info = image_dataset_from_directory(dataset_path / 'train', IMAGE_SIZE,
                                                           augmentation=args.data_augmentation)
    val_ds, _ = image_dataset_from_directory(dataset_path / 'validation', IMAGE_SIZE, shuffle=False)

    # prepare training stuff
    loss = get_loss()
    metrics = get_metrics(train_ds_info)
    callbacks = get_callbacks(checkpoints_path, logs_path)
    class_weights = get_class_weights(train_ds_info) if args.class_weights else None

    # train whole model from scratch
    model = get_model(args.architecture, args.weights, train_ds_info)
    history = model.compile_and_fit(args.learning_rate, loss, metrics, train_ds, val_ds, args.epochs,
                                    args.initial_epoch, callbacks, class_weights)
    model.save_weights(str(models_path / 'model'))
    plot_learning_curves(history, save_path=plots_path)

    if args.pretraining:
        # replace last layer (so that the weights can be loaded for COVID-19 detection) and save
        model.classifier[-1] = Dense(3)
        model.save_weights(str(models_path / 'model_pretrained'))


if __name__ == '__main__':
    main()
