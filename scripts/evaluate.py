import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from covid19.datasets import image_dataset_from_directory
from covid19.metrics import plot_confusion_matrix, plot_roc, make_classification_report
from covid19.models import ResNet50

IMAGE_SIZE = (224, 224)
VERBOSE = 2


def evaluate(model, dataset, output_path, class_names, covid19_label):
    probabilities = model.predict(dataset, verbose=VERBOSE)
    predictions = np.argmax(probabilities, axis=1)
    labels = dataset.classes

    n_classes = len(dataset.class_indices)
    labels_one_hot = tf.one_hot(labels, n_classes)
    covid19_probabilities = probabilities[:, covid19_label]
    covid19_binary_labels = labels_one_hot[:, covid19_label]

    plot_confusion_matrix(labels, predictions, class_names, save_path=output_path)
    plot_roc(covid19_binary_labels, covid19_probabilities, save_path=output_path)
    make_classification_report(labels, predictions, class_names, save_path=output_path)


def main():
    # command-line arguments
    parser = argparse.ArgumentParser(description='Train classifier on COVIDx dataset.')
    parser.add_argument('data', type=str, help='Path to COVIDx dataset')
    parser.add_argument('output', type=str, help='Path where to save the results')
    parser.add_argument('model', type=str, help='Path to the model')
    args = parser.parse_args()

    # prepare paths
    dataset_path = Path(args.data) / 'test'
    output_path = Path(args.output)
    model_path = Path(args.model)
    output_path.mkdir(parents=True, exist_ok=True)

    # build input pipeline
    test_ds = image_dataset_from_directory(dataset_path, IMAGE_SIZE, shuffle=False)
    class_names = sorted(test_ds.class_indices.keys())
    covid19_label = test_ds.class_indices['covid-19']

    # load model
    model = ResNet50()
    model.load_weights(str(model_path))

    # evaluate
    evaluate(model, test_ds, output_path, class_names, covid19_label)


if __name__ == '__main__':
    main()
