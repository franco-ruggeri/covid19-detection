import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """
    Plots the confusion matrix. The confusion matrix is annotated with the absolute counts, while colored according
    to the normalized values by row. In this way, even for imbalanced datasets, the diagonal is highlighted well if the
    predictions are good.

    :param labels: numpy array of shape (n_samples,), ground truth (correct labels)
    :param predictions: numpy array of shape (n_samples,), predictions
    :param class_names: list of class names to use as ticks
    :param save_path: path to the directory where to save the figure (with name 'confusion_matrix.png')
    """
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm / cm.sum(axis=1).reshape(-1, 1)

    plt.figure()
    sns.heatmap(cm_normalized, annot=cm, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap='Reds')
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path / 'confusion_matrix.png')
    plt.close()


def plot_roc(labels, probabilities, save_path=None):
    """
    Plots the ROC curve for binary classification. In case of multi-class labels, the labels should be binarized
    with the respect to a particular class (e.g. labels[:, class_index]).

    :param labels: numpy array of shape (n_samples,), ground truth (correct labels, 1 or 10 or 1)
    :param probabilities: numpy array of shape (n_samples,), probabilities (e.g. softmax activations)
    :param save_path: path to the directory where to save the figure (with name 'roc.png')
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
    plt.legend()
    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(save_path / 'roc.png')
    plt.close()


def make_classification_report(labels, predictions, class_names, save_path=None):
    """
    Generates a classification report including: precision, recall, f1-score, accuracy,

    :param labels: numpy array of shape (n_samples,), ground truth (correct labels)
    :param predictions: numpy array of shape (n_samples,), predictions
    :param class_names: list of class names to use as ticks
    :param save_path: path to the directory where to save the report (with name 'classification_report.txt')
    """
    save_path = Path(save_path)
    cr = classification_report(labels, predictions, target_names=class_names)
    print(cr)
    if save_path is not None:
        save_path = Path(save_path)
        with open(save_path / 'classification_report.txt', mode='w') as f:
            f.write(cr)
