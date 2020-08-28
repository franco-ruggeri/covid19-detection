import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc


def plot_learning_curves(history, history_ft=None, save_path=None):
    """
    Plots the learning curves for all the metrics contained in history (one figure for each metric).

    :param history: History object returned by tf.keras.Model.fit()
    :param save_path: path where to save the figures
    :param history_ft: complete History object returned by tf.keras.Model.fit() including the fine-tuning epochs. If
        this argument is provided, history must contain the History object of without the fine-tuning epochs.
    """
    save_path = Path(save_path)
    epochs = history.epoch[-1]

    for metric in history.history.keys():
        if metric.startswith('val_'):
            continue    # already taken into account as dual metric of another one

        val_metric = 'val_' + metric
        train_values = history.history[metric]
        val_values = history.history[val_metric]
        if history_ft is not None:
            train_values += history_ft.history[metric]
            val_values += history_ft.history[val_metric]

        plt.figure()
        plt.plot(train_values, label='training')
        plt.plot(val_values, label='validation')
        if history_ft is not None:
            plt.plot([epochs, epochs], plt.ylim(), label='start fine-tuning')
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.show()
        if save_path is not None:
            plt.savefig(save_path / (metric + '.png'))


def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """
    Plots the confusion matrix.

    :param labels: numpy array of shape (n_samples,), ground truth (correct labels)
    :param predictions: numpy array of shape (n_samples,), predictions
    :param class_names: list of class names to use as ticks
    :param save_path: path where to save the figure
    """
    cm = confusion_matrix(labels, predictions, normalize='true')
    plt.figure()
    sns.heatmap(cm, annot=True, fmt=".1%", xticklabels=class_names, yticklabels=class_names, cmap='Reds')
    plt.ylabel('Ground truth')
    plt.xlabel('Predicted label')
    plt.show()
    if save_path is not None:
        plt.savefig(save_path / 'confusion_matrix.png')


def plot_roc(labels, probabilities, save_path=None):
    """
    Plots the ROC curve for binary classification. In case of multi-class labels, the labels should be binarized
    with the respect to a particular class (e.g. labels[:, class_index]).

    :param labels: numpy array of shape (n_samples,), ground truth (correct labels, 1 or 10 or 1)
    :param probabilities: numpy array of shape (n_samples,), probabilities (e.g. softmax activations)
    :param save_path: path where to save the figure
    """
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area={:.2f})'.format(roc_auc), linewidth=2)
    plt.plot([0, 1], [0, 1], label='random choice', linestyle='--', linewidth=2)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path / 'roc.png')


def make_classification_report(labels, predictions, class_names, save_path):
    """
    Generates a classification report including: precision, recall, f1-score, accuracy,

    :param labels:
    :param predictions:
    :param class_names:
    :param save_path:
    """
    save_path = Path(save_path)
    cr = classification_report(labels, predictions, target_names=class_names)
    with open(save_path / 'classification_report.txt', mode='w') as f:
        f.write(cr)
