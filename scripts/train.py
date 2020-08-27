import argparse
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

IMAGE_SHAPE = (224, 224)
INPUT_SHAPE = IMAGE_SHAPE + (3,)
AUTO_TUNE = tf.data.experimental.AUTOTUNE


def get_dataset(dataset_path):
    dataset_get = ImageDataGenerator()
    dataset = dataset_get.flow_from_directory(dataset_path, target_size=IMAGE_SHAPE)
    return dataset


def plot_learning_curves(history, history_ft=None):
    if history_ft is not None:
        history.history['accuracy'] += history_ft.history['accuracy']
        history.history['loss'] += history_ft.history['loss']
        history.history['val_accuracy'] += history_ft.history['val_accuracy']
        history.history['val_loss'] += history_ft.history['val_loss']

    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='training')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    if history_ft is not None:
        plt.plot([history.epoch[-1], history.epoch[-1]], plt.ylim(), label='start fine-tuning')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if history_ft is not None:
        plt.plot([history.epoch[-1], history.epoch[-1]], plt.ylim(), label='start fine-tuning')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser(description='Train classifier on COVIDx dataset.')
    parser.add_argument('--data', type=str, help='Path to COVIDx dataset', required=True)
    parser.add_argument('--model', type=str, help='Path where to save the trained model', required=True)
    parser.add_argument('--logs', type=str, help='Path where to save logs for TensorBoard', default='logs')
    args = parser.parse_args()

    # prepare paths
    dataset_path = Path(args.data)
    model_path = Path(args.model)
    logs_path = Path(args.logs)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    # build input pipeline
    train_ds = get_dataset(dataset_path / 'train')
    val_ds = get_dataset(dataset_path / 'validation')
    test_ds = get_dataset(dataset_path / 'test')
    n_classes = train_ds.num_classes

    # plot one image (for debugging)
    images = next(iter(train_ds))[0]
    for image in images[:10]:
        image = preprocess_input(image)
        window_name = 'Example'
        cv2.imshow(window_name, image)
        cv2.waitKey(1000)
        cv2.destroyWindow(window_name)

    # build model
    base = ResNet50V2(include_top=False, pooling='avg', input_shape=INPUT_SHAPE)
    head = Dense(n_classes)
    inputs = Input(shape=INPUT_SHAPE)
    x = preprocess_input(inputs)
    x = base(x, training=False)
    outputs = head(x)
    model = Model(inputs, outputs)

    # training settings
    learning_rate = 0.001
    epochs = 20
    loss = CategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']
    callbacks = [
        ModelCheckpoint(filepath=model_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=2),
        # TensorBoard(log_dir=logs_path)
    ]

    # train only head (feature extraction with pre-trained model)
    base.trainable = False
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)
    model.summary()
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
    plt.figure()
    plot_learning_curves(history)

    # fine-tune some layers
    learning_rate /= 10
    epochs += 10
    fine_tune_at = 20
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics=metrics)
    model.summary()
    history_ft = model.fit(train_ds, epochs=epochs, initial_epoch=history.epoch[-1]+1, validation_data=val_ds,
                           callbacks=callbacks)
    plot_learning_curves(history, history_ft)
    model.save(model_path)

    # evaluate model
    # TODO: night's test should give over-fitting (>90% training accuracy, poor generalization)
    # TODO: add confusion matrix and other metrics for evaluation (code below) -> should suck now
    # TODO: add data augmentation -> over-fitting should be reduced (i.e. test accuracy should improve)
    # TODO: add resampling -> confusion matrix should improve
    # TODO: add metric for cross-validation, val_acc is not adequate due to the imbalance (e.g. f1 score, but we have
    #  3 classes..., google it)
    # TODO: add grad-cam
    # TODO: add COVID-Net
    # TODO: run tests
    # TODO: image saving and model loading

    # confusion matrix
    # cm = confusion_matrix(test_data_gen.classes, predictions)
    # ticks = np.arange(n_classes)
    # plt.figure()
    # sns.heatmap(cm, annot=True, fmt="d")
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # plt.savefig(os.path.join(args.results, model_name + '_confusion_matrix.png'))
    # plt.show()
    #
    # # precision, recall, f1-score, accuracy, etc.
    # cr = classification_report(test_data_gen.classes, predictions, target_names=class_names)
    # print('Classification report')
    # print(cr)
    # with open(os.path.join(args.results, model_name + '_report.txt'), mode='w') as f:
    #     f.write(cr)
