import argparse
import numpy as np
from pathlib import Path
from covid19.datasets import image_dataset_from_directory
from covid19.metrics import plot_learning_curves
from covid19.models import ResNet50
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.metrics import CategoricalAccuracy, AUC, Precision, Recall
from tensorflow_addons.metrics import F1Score

# training settings
LR = 0.0001
LR_FT = LR / 10             # learning rate for fine-tuning
EPOCHS = 30
EPOCHS_FT = 10              # epochs for fine-tuning
FINE_TUNE_AT = 150          # layer at which to start fine-tuning (layers [0, fine_tune_at-1] are frozen)

IMAGE_SHAPE = (224, 224, 3)
VERBOSE = 2


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
        AUC(name='auc', multi_label=True),  # multi_label=True => macro-average
        F1Score(name='f1-score', num_classes=n_classes, average='macro'),
        Precision(name='precision_covid19', class_id=covid19_label),
        Recall(name='recall_covid19', class_id=covid19_label),
    ]


def get_callbacks(checkpoints_path, logs_path):
    filepath_checkpoint = checkpoints_path / 'epoch_{epoch:02d}'
    return [
        ModelCheckpoint(filepath=str(filepath_checkpoint), save_weights_only=True, verbose=VERBOSE),
        EarlyStopping(patience=5, restore_best_weights=True, verbose=VERBOSE),
        TensorBoard(log_dir=logs_path, profile_batch=0)
    ]


def get_class_weights(train_ds, train_ds_info):
    total = train_ds_info['n_images']
    n_classes = train_ds['n_classes']
    class_weights = {}

    for class_name, class_label in train_ds_info['class_labels'].items():
        # scale weights by total / n_classes to keep the loss to a similar magnitude
        # see https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
        n = len(np.where(train_ds.labels == class_label)[0])
        class_weights[class_label] = (1 / n) * (total / n_classes)
    return class_weights


def main():
    # command-line arguments
    parser = argparse.ArgumentParser(description='Train COVID-19 classifier.')
    parser.add_argument('data', type=str, help='Path to COVIDx dataset')
    parser.add_argument('model', type=str, help='Path where to save trained model, checkpoints and logs. Must be a '
                                                'non-existing directory.')
    parser.add_argument('--class-weights', action='store_true', default=False, help='Use class weights to compensate '
                                                                                    'the dataset imbalance.')
    parser.add_argument('--data-augmentation', action='store_true', default=False, help='Augment data during training')
    args = parser.parse_args()

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
    train_ds, train_ds_info = image_dataset_from_directory(dataset_path / 'train', IMAGE_SHAPE[0:2],
                                                           augmentation=args.data_augmentation)
    val_ds, _ = image_dataset_from_directory(dataset_path / 'validation', IMAGE_SHAPE[0:2], shuffle=False)

    # prepare training stuff
    loss = get_loss()
    metrics = get_metrics(train_ds_info)
    callbacks = get_callbacks(checkpoints_path, logs_path)
    class_weights = get_class_weights(train_ds) if args.class_weights else None

    # train model
    model = ResNet50()
    history = model.fit_classifier(LR, loss, metrics, train_ds, val_ds, EPOCHS, 0, callbacks, class_weights)
    model.save_weights(str(models_path / 'model_no_ft'))
    history_ft = model.fine_tune(LR_FT, loss, metrics, train_ds, val_ds, EPOCHS_FT, history.epoch[-1]+1, callbacks,
                                 FINE_TUNE_AT, class_weights)
    model.save_weights(str(models_path / 'model_ft'))
    plot_learning_curves(history, history_ft, save_path=plots_path)


if __name__ == '__main__':
    main()


# TODO (tests):
#   1) with vs without class weights -> class weights gives better confusion matrix, continue using it
#   2) fine-tune better (smaller learning rate and less layers)
#   3) class weights + data augmentation, does the generalization improve? if so, continue using it
#   4) class weights + no pre-training [+ data augmentation], should be worse than pre-trained

# TODO (code):
#   1) fix COVID-Net
