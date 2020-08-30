import argparse
from pathlib import Path
from covid19.metrics import plot_learning_curves
from covid19.preprocessing import image_balanced_dataset_from_directory, image_dataset_from_directory
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.metrics import CategoricalAccuracy, AUC, Precision, Recall
from tensorflow_addons.metrics import F1Score

# training settings
BATCH_SIZE = 32
LR = 0.0001
LR_FT = LR / 10             # learning rate for fine-tuning
EPOCHS = 20
EPOCHS_FT = 10              # epochs for fine-tuning
FINE_TUNE_AT = 150          # layer at which to start fine-tuning (layers [0, fine_tune_at-1] are frozen)

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = IMAGE_SIZE + (3,)
VERBOSE = 2


def make_model(n_classes):
    feature_extractor = ResNet50V2(include_top=False, pooling='avg', input_shape=INPUT_SHAPE)
    inputs = Input(shape=INPUT_SHAPE)
    x = feature_extractor(inputs, training=False)           # training=False to keep BN layers in inference mode
    outputs = Dense(n_classes, activation='softmax')(x)     # softmax necessary for AUC metric
    return Model(inputs=inputs, outputs=outputs, name='covidnet')


def get_loss():
    return CategoricalCrossentropy()


def get_metrics(n_classes, covid19_label):
    # remarks:
    # - AUC and F1-score are computed with macro-average (we care a lot about the small COVID-19 class!)
    # - precision and recall are computed only on the COVID-19 class (again, it is the most important)
    return [
        CategoricalAccuracy(name='accuracy'),
        AUC(name='auc', multi_label=True),  # multi_label=True => macro-average
        F1Score(name='f-score', num_classes=n_classes, average='macro'),
        Precision(name='precision', class_id=covid19_label),
        Recall(name='recall', class_id=covid19_label),
    ]


def get_callbacks(model_path, logs_path):
    return [
        ModelCheckpoint(filepath=model_path, monitor='val_auc', mode='max', save_best_only=True, verbose=VERBOSE),
        EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=VERBOSE),
        TensorBoard(log_dir=logs_path, profile_batch=0)
    ]


def train(model, train_ds, val_ds, learning_rate, epochs, initial_epoch, loss, metrics, callbacks, fine_tune=False):
    # set trainable layers
    feature_extractor = model.layers[-2]
    feature_extractor.trainable = False
    if fine_tune:
        feature_extractor.trainable = True      # unfreeze convolutional base
        for layer in feature_extractor.layers[:FINE_TUNE_AT]:
            layer.trainable = False             # freeze bottom layers

    # compile and fit
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)
    model.summary()
    return model.fit(train_ds, epochs=epochs+initial_epoch, initial_epoch=initial_epoch,
                     steps_per_epoch=train_ds.n_batches, validation_data=val_ds, callbacks=callbacks)


if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser(description='Train classifier on COVIDx dataset.')
    parser.add_argument('data', type=str, help='Path to COVIDx dataset')
    parser.add_argument('model', type=str, help='Path where to save the trained model and the logs. Must be a '
                                                'non-existing path. The path will be created.')
    args = parser.parse_args()

    # prepare paths
    dataset_path = Path(args.data)
    model_path = Path(args.model)
    if model_path.is_dir():
        raise FileExistsError(str(model_path) + ' already exists')
    model_path.mkdir(parents=True)
    logs_path = model_path / 'logs'
    plots_path = model_path / 'training'
    model_path = model_path / 'model.h5'
    plots_path.mkdir()

    # build input pipeline
    train_ds = image_balanced_dataset_from_directory(dataset_path / 'train', IMAGE_SIZE, BATCH_SIZE)
    val_ds = image_dataset_from_directory(dataset_path / 'validation', IMAGE_SIZE, BATCH_SIZE, shuffle=False)
    n_classes = len(train_ds.class_indices)
    covid19_label = train_ds.class_indices['covid-19']

    # compose model
    model = make_model(n_classes)

    # train top layer and fine-tune
    loss = get_loss()
    metrics = get_metrics(n_classes, covid19_label)
    callbacks = get_callbacks(model_path, logs_path)
    history = train(model, train_ds, val_ds, LR, EPOCHS, 0, loss, metrics, callbacks)
    model.save(model_path.with_name(model_path.stem + '_no_finetuning' + model_path.suffix))
    history_ft = train(model, train_ds, val_ds, LR_FT, EPOCHS_FT, history.epoch[-1] + 1, loss, metrics,
                       callbacks, fine_tune=True)
    model.save(model_path)
    plot_learning_curves(history, history_ft, save_path=plots_path)


# TODO: test early stopping (history.epoch[-1] contains the correct number of epochs?)

# TODO: run with and without resampling, confusion matrix should improve
# TODO: add data augmentation -> over-fitting should be reduced (i.e. test accuracy should improve)
# TODO: add grad-cam
# TODO: add COVID-Net and train the 3 models as resnet50
# TODO: add histogram for dataset

# TODO: se ottengo risultati di merda puo' essere che la pipeline nuova e' sbagliata
# TODO: subclass models to have a simple fit() method including fine-tuning, so that the package is easily usable
