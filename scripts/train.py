import argparse
import covid19.utils
from pathlib import Path


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
        covid19.utils.train(dataset_path, model_path, logs_path)
    covid19.utils.evaluate(dataset_path, model_path)


# TODO: add data augmentation -> over-fitting should be reduced (i.e. test accuracy should improve)
# TODO: add resampling -> confusion matrix should improve
# TODO: add metric for cross-validation, val_acc is not adequate due to the imbalance (e.g. f1 score, but we have
#  3 classes..., google it)
# TODO: save models at each epoch, not only the best
# TODO: add grad-cam
# TODO: add COVID-Net
# TODO: run tests
