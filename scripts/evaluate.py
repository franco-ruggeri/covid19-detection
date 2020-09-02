import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from covid19.datasets import image_dataset_from_directory
from covid19.metrics import plot_confusion_matrix, plot_roc, make_classification_report
from covid19.models import ResNet50
from covid19.explainers import GradCAM, plot_explanation

VERBOSE = 2


def evaluate(model, dataset, output_path):
    probabilities = model.predict(dataset, verbose=VERBOSE)
    predictions = np.argmax(probabilities, axis=1)
    labels = dataset.classes

    class_names = sorted(dataset.class_indices.keys())
    covid19_label = dataset.class_indices['covid-19']
    n_classes = len(class_names)
    labels_one_hot = tf.one_hot(labels, n_classes)
    covid19_probabilities = probabilities[:, covid19_label]
    covid19_binary_labels = labels_one_hot[:, covid19_label]

    plot_confusion_matrix(labels, predictions, class_names, save_path=output_path)
    plot_roc(covid19_binary_labels, covid19_probabilities, save_path=output_path)
    make_classification_report(labels, predictions, class_names, save_path=output_path)


def explain(model, dataset, output_path):
    grad_cam = GradCAM(model)
    class_labels = dataset.class_indices
    count = {label: 0 for label in class_labels.values()}
    class_names = {label: class_name for class_name, label in class_labels.items()}    # reverse
    n_batches = len(dataset)

    with tqdm(total=n_batches * dataset.batch_size) as bar:
        bar.set_description('Explaining images in ' + str(output_path))

        for _ in range(n_batches):   # can't directly iterate over dataset, as it iterates forever
            batch = next(dataset)
            images = batch[0]
            labels = batch[1]

            for image, label in zip(images, labels):
                bar.update()

                prediction, explanation = grad_cam.explain(image)
                label = np.argmax(label)
                save_path = output_path / (class_names[label] + '_{:05d}'.format(count[label]) + '.png')
                title = 'Prediction: {}\nGround truth: {}'.format(class_names[prediction], class_names[label])
                plot_explanation(image, explanation, title=title, save_path=save_path)
                count[label] += 1


def main():
    # command-line arguments
    parser = argparse.ArgumentParser(description='Test COVID-19 classifier.')
    parser.add_argument('analysis', type=str, help='Type of evaluation. Support for: performance, explainability.')
    parser.add_argument('data', type=str, help='Path to COVIDx dataset')
    parser.add_argument('output', type=str, help='Path where to save the results')
    parser.add_argument('model', type=str, help='Path to the model')
    args = parser.parse_args()

    # prepare paths
    dataset_path = Path(args.data) / 'test'
    output_path = Path(args.output)
    model_path = Path(args.model)
    output_path.mkdir(parents=True, exist_ok=True)

    # load model
    model = ResNet50()
    model.load_weights(str(model_path))

    # build input pipeline
    test_ds = image_dataset_from_directory(dataset_path, model.image_shape[0:2], shuffle=False)

    # evaluate
    if args.analysis == 'performance':
        evaluate(model, test_ds, output_path)
    elif args.analysis == 'explainability':
        explain(model, test_ds, output_path)
    else:
        raise ValueError('Invalid command')


if __name__ == '__main__':
    main()
