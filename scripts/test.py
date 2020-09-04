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


def evaluate(model, dataset, dataset_info, output_path):
    probabilities = model.predict(dataset, verbose=VERBOSE)
    predictions = np.argmax(probabilities, axis=1)
    labels = dataset_info['labels']
    labels_one_hot = tf.one_hot(labels, dataset_info['n_classes'])

    class_names = sorted(dataset_info['class_labels'].keys())
    covid19_label = dataset_info['class_labels']['covid-19']
    covid19_probabilities = probabilities[:, covid19_label]
    covid19_binary_labels = labels_one_hot[:, covid19_label]

    plot_confusion_matrix(labels, predictions, class_names, save_path=output_path)
    plot_roc(covid19_binary_labels, covid19_probabilities, save_path=output_path)
    make_classification_report(labels, predictions, class_names, save_path=output_path)


def explain(model, dataset, dataset_info, output_path):
    grad_cam = GradCAM(model)
    count = {label: 0 for label in dataset_info['class_labels'].values()}
    class_names = {label: class_name for class_name, label in dataset_info['class_labels'].items()}    # reverse
    n_batches = dataset_info['n_batches']
    iter_dataset = iter(dataset)

    with tqdm(total=n_batches * dataset_info['batch_size']) as bar:
        bar.set_description('Explaining images')

        for _ in range(n_batches):   # can't directly iterate over dataset, as it iterates forever
            batch = next(iter_dataset)
            images = batch[0]
            labels = batch[1]

            for image, label in zip(images, labels):
                bar.update()

                prediction, explanation = grad_cam.explain(image)
                label = np.argmax(label)
                save_path = output_path / (class_names[label] + '_{:05d}'.format(count[label]) + '.png')
                plot_explanation(image, explanation, class_names[prediction], class_names[label], save_path=save_path)
                count[label] += 1


def main():
    # command-line arguments
    parser = argparse.ArgumentParser(description='Test COVID-19 detection model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('analysis', type=str, help='type of evaluation. Supported: performance, explainability.')
    parser.add_argument('data', type=str, help='path to COVIDx dataset')
    parser.add_argument('output', type=str, help='path where to save the results')
    parser.add_argument('model', type=str, help='path to the model')
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
    test_ds, test_ds_info = image_dataset_from_directory(dataset_path, model.image_shape[0:2], shuffle=False)

    # evaluate
    if args.analysis == 'performance':
        evaluate(model, test_ds, test_ds_info, output_path)
    elif args.analysis == 'explainability':
        explain(model, test_ds, test_ds_info, output_path)
    else:
        raise ValueError('Invalid analysis')


if __name__ == '__main__':
    main()
