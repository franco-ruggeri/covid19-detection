import argparse
import numpy as np
from pathlib import Path
from covid19.metrics import plot_learning_curves, plot_confusion_matrix, plot_roc, make_classification_report
from covid19.preprocessing import image_balanced_dataset_from_directory, image_dataset_from_directory
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import CategoricalAccuracy, AUC, Precision, Recall
from tensorflow_addons.metrics import F1Score

# training settings
BATCH_SIZE = 32
LR = 0.0001
LR_FT = LR / 10             # learning rate for fine-tuning
EPOCHS = 30
EPOCHS_FT = EPOCHS + 10     # total epochs including fine-tuning (not just fine-tuning)
FINE_TUNE_AT = 100          # layer at which to start fine-tuning (layers [0, fine_tune_at-1] are frozen)

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = IMAGE_SIZE + (3,)
VERBOSE = 1


def train(dataset_path, model_path, logs_path):
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    plots_path = model_path.parent

    # build input pipeline
    train_ds, classes, n_images = image_balanced_dataset_from_directory(dataset_path / 'train', IMAGE_SIZE, BATCH_SIZE)
    val_ds, _ = image_dataset_from_directory(dataset_path / 'validation', IMAGE_SIZE, BATCH_SIZE, shuffle=False)
    steps_per_epoch = n_images / BATCH_SIZE
    n_classes = len(classes)
    covid19_label = train_ds.class_indixes['covid-19']

    # build model
    feature_extractor = ResNet50V2(include_top=False, pooling='avg', input_shape=INPUT_SHAPE)
    inputs = Input(shape=INPUT_SHAPE)
    x = preprocess_input(inputs)
    x = feature_extractor(x, training=False)    # training=False to keep BN layers in inference mode when unfrozen
    outputs = Dense(n_classes, activation='softmax')(x)     # softmax necessary for AUC metric
    model = Model(inputs, outputs)

    # define loss, metrics, and callbacks
    # remarks:
    # - AUC and F1-score are computed with macro-average (we care a lot about the small COVID-19 class!)
    # - precision and recall are computed only on the COVID-19 class (again, it is the most important)
    loss = CategoricalCrossentropy()
    metrics = [
        CategoricalAccuracy(name='accuracy'),
        AUC(name='auc', multi_label=True),      # multi_label=True => macro-average
        F1Score(name='f-score', num_classes=n_classes, average='macro'),
        Precision(name='precision', class_id=covid19_label),
        Recall(name='recall', class_id=covid19_label),
    ]
    callbacks = [
        ModelCheckpoint(filepath=model_path.with_name(model_path.stem + '-{epoch:02d}-{val_auc:.2f}-{val_f-score:.2f}' +
                                                      model_path.suffix), verbose=VERBOSE),
        TensorBoard(log_dir=logs_path, profile_batch=0)
    ]

    # train only top layer (feature extraction with pre-trained convolutional base)
    feature_extractor.trainable = False
    model.compile(optimizer=Adam(lr=LR), loss=loss, metrics=metrics)
    model.summary()
    history = model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=val_ds,
                        callbacks=callbacks)
    plot_learning_curves(history, save_path=plots_path)

    # fine-tune some layers
    feature_extractor.trainable = True
    for layer in feature_extractor.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    model.compile(optimizer=Adam(lr=LR_FT), loss=loss, metrics=metrics)
    model.summary()
    history_ft = model.fit(train_ds, epochs=EPOCHS_FT, initial_epoch=EPOCHS, steps_per_epoch=steps_per_epoch,
                           validation_data=val_ds, callbacks=callbacks)
    plot_learning_curves(history, history_ft=history_ft, save_path=plots_path)
    model.save(model_path)


def evaluate(dataset_path, model_path):
    model_path = Path(model_path)
    plots_path = model_path.parent

    model = load_model(model_path)
    test_ds, class_indexes = image_dataset_from_directory(dataset_path / 'test', IMAGE_SIZE, BATCH_SIZE, shuffle=False)
    class_names = sorted(class_indexes.keys())
    covid19_index = class_indexes['covid-19']

    probabilities = model.predict(test_ds)
    predictions = np.argmax(probabilities, axis=1)
    labels_one_hot = np.array([label.numpy() for _, label in test_ds.unbatch()])
    labels = np.argmax(labels_one_hot, axis=1)
    covid19_probabilities = probabilities[:, covid19_index]
    covid19_binary_labels = labels_one_hot[:, covid19_index]

    plot_confusion_matrix(labels, predictions, class_names, save_path=plots_path)
    plot_roc(covid19_binary_labels, covid19_probabilities, save_path=plots_path)
    make_classification_report(labels, predictions, class_names, save_path=plots_path)


if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser(description='Train classifier on COVIDx dataset.')
    parser.add_argument('data', type=str, help='Path to COVIDx dataset')
    parser.add_argument('models', type=str, help='Path where to save the trained model')
    parser.add_argument('model', type=str, help='Model name. If a file <model>.h5 exists, the model will be loaded and '
                                                'just evaluated. Otherwise, it will be trained, saved and evaluated.')
    parser.add_argument('--logs', type=str, default='logs', help='Path where to save logs for TensorBoard')
    args = parser.parse_args()

    # prepare paths
    dataset_path = Path(args.data)
    model_path = Path(args.models) / args.model
    model_path = model_path.with_suffix('.h5')
    logs_path = Path(args.logs)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    # train and evaluate model
    if not model_path.is_file():
        train(dataset_path, model_path, logs_path)
    evaluate(dataset_path, model_path)


# TODO: debugga evaluate(), confusion matrix di merda, confronta col dataset di prima
# TODO: run with and without resampling, confusion matrix should improve
# TODO: add data augmentation -> over-fitting should be reduced (i.e. test accuracy should improve)
# TODO: add grad-cam
# TODO: add COVID-Net and train the 3 models as resnet50
# TODO: add histogram for dataset

# TODO: if still overfitting... add early stopping
# TODO: subclass models to have a simple fit() method including fine-tuning, so that the package is easily usable
