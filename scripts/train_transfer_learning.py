#!/usr/bin/env python3

import argparse
from utils import get_model, get_callbacks, get_class_weights, IMAGE_SIZE
from pathlib import Path
from covid19.datasets import image_dataset_from_directory
from covid19.metrics import plot_learning_curves
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC, Precision, Recall
from tensorflow_addons.metrics import F1Score


def get_command_line_arguments():
    parser = argparse.ArgumentParser(description='Train COVID-19 detection model with transfer learning.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=str, help='path to COVIDx dataset')
    parser.add_argument('model', type=str,
                        help='path where to save trained model, checkpoints and logs. Must be a non-existing '
                             'directory.')
    parser.add_argument('weights', type=str, help='`imagenet` (only for resnet50), or path to the pretrained weights')
    parser.add_argument('--architecture', type=str, default='resnet50',
                        help='architecture to use. Supported: resnet50, covidnet.')
    parser.add_argument('--class-weights', action='store_true', default=False,
                        help='compensate dataset imbalance using class weights')
    parser.add_argument('--data-augmentation', action='store_true', default=False, help='augment data during training')
    parser.add_argument('--initial-epoch', type=int, default=0,
                        help='initial epochs to skip (useful for resuming training)')
    parser.add_argument('--load-weights', type=str, default=None,
                        help='path to weights to be loaded (useful for resuming training)')
    parser.add_argument('--epochs', type=int, default=30, help='epochs of training for classifier on top')
    parser.add_argument('--epochs-ft', type=int, default=10, help='epochs of fine-tuning')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='learning rate for training classifier on top')
    parser.add_argument('--learning-rate-ft', type=float, default=1e-6, help='learning rate for fine-tuning')
    parser.add_argument('--fine-tune-at', type=int, default=0, help='index of layer at which to start to unfreeze')
    return parser.parse_args()


def get_loss():
    return CategoricalCrossentropy()


def get_metrics(dataset_info):
    # remarks:
    # - AUC and F1-score are computed with macro-average (we care a lot about the small COVID-19 class!)
    # - precision and recall are computed only on the COVID-19 class (again, it is the most important)
    n_classes = dataset_info['n_classes']
    covid19_label = dataset_info['class_labels']['covid-19']
    return [
        CategoricalAccuracy(name='accuracy'),
        AUC(name='auc', multi_label=True),      # multi_label=True => macro-average
        F1Score(name='f1-score', num_classes=n_classes, average='macro'),
        Precision(name='precision_covid19', class_id=covid19_label),
        Recall(name='recall_covid19', class_id=covid19_label),
    ]


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

    # train linear classifier
    model = get_model(args.architecture, args.weights, train_ds_info, args.load_weights)
    history = model.fit_linear_classifier(args.learning_rate, loss, metrics, train_ds, val_ds, args.epochs,
                                          args.initial_epoch, callbacks, class_weights)
    model.save_weights(str(models_path / 'model_no_ft'))

    # fine-tune some layers
    history_ft = model.fine_tune(args.learning_rate_ft, loss, metrics, train_ds, val_ds, args.epochs_ft,
                                 args.initial_epoch + args.epochs, callbacks, args.fine_tune_at, class_weights)
    model.save_weights(str(models_path / 'model_ft'))
    plot_learning_curves(history, history_ft, save_path=plots_path)


if __name__ == '__main__':
    main()
