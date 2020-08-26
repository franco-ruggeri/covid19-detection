import argparse
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

IMAGE_SHAPE = (224, 224)
INPUT_SHAPE = IMAGE_SHAPE + (3,)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_dataset(directory):
    dataset = image_dataset_from_directory(directory, label_mode='categorical')
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser(description='Train classifier on COVIDx dataset.')
    parser.add_argument('--dataset', type=str, help='Path to COVIDx dataset', required=True)
    args = parser.parse_args()

    # prepare paths
    dataset_path = Path(args.dataset)
    train_path = dataset_path / 'train'
    val_path = dataset_path / 'validation'
    test_path = dataset_path / 'test'

    # build input pipeline
    train_ds = get_dataset(train_path)
    val_ds = get_dataset(val_path)
    test_ds = get_dataset(test_path)

    # plot one batch
    # TODO: this does not work, check return types, plot one image, take the number of classes
    batch = train_ds.take(1)
    image, label = next(iter(batch))
    print(type(image))
    print(label)
    cv2.imshow('example', image.numpy)
    cv2.waitKey()
    n_classes = len(label)

    # build model
    base = ResNet50V2(include_top=False, pooling='avg', input_shape=INPUT_SHAPE)
    head = Dense(n_classes)
    inputs = Input(shape=INPUT_SHAPE)
    x = preprocess_input(inputs)
    x = base(x, training=False)
    outputs = head(x)
    model = Model(inputs, outputs)
    model.summary()

    # train model
    learning_rate = 0.001
    epochs = 10
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

    # evaluate model
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='training')
    plt.plot(val_acc, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='training')
    plt.plot(val_loss, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
