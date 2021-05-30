import numpy as np
from pathlib import Path
from covid19.datasets import image_dataset_from_directory
from covid19.metrics import plot_learning_curves
from covid19.cli._utils import get_model, get_image_size
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.metrics import CategoricalAccuracy, AUC, Precision, Recall
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.layers import Dense


def _add_arguments_common(parser):
    parser.add_argument('--architecture', type=str, default='resnet50', choices=['resnet50', 'covidnet'],
                        help='architecture to use.')
    parser.add_argument('--class-weights', action='store_true', default=False,
                        help='compensate dataset imbalance using class weights')
    parser.add_argument('--data-augmentation', action='store_true', default=False, help='augment data during training')
    parser.add_argument('--load-weights', type=str, default=None,
                        help='path to weights to be loaded (useful for resuming training)')
    parser.add_argument('--initial-epoch', type=int, default=3,
                        help='initial epochs to skip (useful for resuming training)')
    parser.add_argument('--epochs', type=int, default=30, help='epochs of training')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--loss', type=str, default='categorical_crossentropy', help='loss function for training')
    parser.add_argument('--verbose', type=int, default=2, help='verbosity level of the output')


def _add_arguments_from_scratch(parser):
    parser.add_argument('data', type=str, help='path to the dataset')
    parser.add_argument('model', type=str,
                        help='path where to save the training output (e.g., trained model), must not exist')
    _add_arguments_common(parser)


def _add_arguments_transfer_learning(parser):
    parser.add_argument('data', type=str, help='path to the dataset')
    parser.add_argument('model', type=str,
                        help='path where to save the training output (e.g., trained model), must not exist')
    parser.add_argument('weights', type=str, help='`imagenet` (only for resnet50), or path to the pretrained weights')
    parser.add_argument('--n-classes-pretrain', type=int, default=3,
                        help='number of classes in the dataset used for pretraining (no need for resnet50 on imagenet.')
    parser.add_argument('--epochs-ft', type=int, default=10, help='epochs of fine-tuning')
    parser.add_argument('--learning-rate-ft', type=float, default=1e-6, help='learning rate for fine-tuning')
    parser.add_argument('--fine-tune-at', type=int, default=0, help='index of layer at which to start to unfreeze')
    _add_arguments_common(parser)


def add_arguments_train(parser):
    subparsers = parser.add_subparsers()
    parser_from_scratch = subparsers.add_parser('from_scratch')
    parser_transfer_learning = subparsers.add_parser('transfer_learning')

    _add_arguments_from_scratch(parser_from_scratch)
    _add_arguments_transfer_learning(parser_transfer_learning)


def _get_metrics(dataset_info):
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


def _get_callbacks(checkpoints_path, logs_path, verbosity):
    filepath_checkpoint = checkpoints_path / 'epoch_{epoch:02d}'
    return [
        ModelCheckpoint(filepath=str(filepath_checkpoint), save_weights_only=True, verbose=verbosity),
        TensorBoard(log_dir=logs_path, profile_batch=0)
    ]


def _get_class_weights(train_ds_info):
    total = train_ds_info['n_images']
    n_classes = train_ds_info['n_classes']
    class_weights = {}

    for class_name, class_label in train_ds_info['class_labels'].items():
        # scale weights by total / n_classes to keep the loss to a similar magnitude
        # see https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
        n = len(np.where(train_ds_info['labels'] == class_label)[0])
        class_weights[class_label] = (1 / n) * (total / n_classes)
    return class_weights


def train(args):
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
    image_size = get_image_size(args.architecture)
    train_ds, train_ds_info = image_dataset_from_directory(dataset_path / 'train', image_size,
                                                           augmentation=args.data_augmentation)
    val_ds, _ = image_dataset_from_directory(dataset_path / 'validation', image_size, shuffle=False)
    n_classes = train_ds_info['n_classes']

    # prepare training stuff
    metrics = _get_metrics(train_ds_info)
    callbacks = _get_callbacks(checkpoints_path, logs_path, args.verbosity)
    class_weights = _get_class_weights(train_ds_info) if args.class_weights else None

    # train
    if args.mode == 'from_scratch':
        # train whole model from scratch
        model = get_model(args.architecture, None, n_classes, args.load_weights)
        history = model.compile_and_fit(
            learning_rate=args.learning_rate,
            loss=args.loss,
            metrics=metrics,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=args.epochs,
            initial_epoch=args.initial_epoch,
            callbacks=callbacks,
            class_weights=class_weights
        )
        model.save_weights(str(models_path / 'model'))
        plot_learning_curves(history, save_path=plots_path)

    elif args.mode == 'transfer_learning':
        # replace last layer with the right number of units
        model = get_model(args.architecture, args.weights, args.n_classes_pretrain, args.load_weights)
        model.classifier.pop()
        model.classifier.add(Dense(n_classes))

        # train linear classifier
        history = model.fit_linear_classifier(
            learning_rate=args.learning_rate,
            loss=args.loss,
            metrics=metrics,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=args.epochs,
            initial_epoch=args.initial_epoch,
            callbacks=callbacks,
            class_weights=class_weights
        )
        model.save_weights(str(models_path / 'model_no_ft'))

        # fine-tune some layers
        history_ft = model.fine_tune(
            learning_rate=args.learning_rate_ft, 
            loss=args.loss, 
            metrics=metrics, 
            train_ds=train_ds, 
            val_ds=val_ds, 
            epochs=args.epochs_ft,
            initial_epoch=args.initial_epoch + args.epochs, 
            callbacks=callbacks, 
            fine_tune_at=args.fine_tune_at, 
            class_weights=class_weights
        )
        model.save_weights(str(models_path / 'model_ft'))
        plot_learning_curves(history, history_ft=history_ft, save_path=plots_path)

    else:
        raise ValueError
