import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_learning_curves(history, history_ft=None, save_path=None):
    """
    Plots the learning curves for all the metrics contained in history (one figure for each metric).

    :param history: History object returned by tf.keras.Model.fit()
    :param history_ft: History object returned by tf.keras.Model.fit() including the fine-tuning epochs. If this
        argument is provided, history must contain the History object of without the fine-tuning epochs.
    :param save_path: path to the directory where to save the figures (with names '<metric>.png')
    """
    epochs = history.epoch[-1] + 1

    for metric in history.history.keys():
        if metric.startswith('val_'):
            continue    # already taken into account as dual metric of another one

        val_metric = 'val_' + metric
        train_values = history.history[metric]
        val_values = history.history[val_metric]
        if history_ft is not None:
            train_values += history_ft.history[metric]
            val_values += history_ft.history[val_metric]

        x = np.arange(len(train_values))
        plt.figure()
        plt.plot(x, train_values, label='training')
        plt.plot(x, val_values, label='validation')
        if history_ft is not None:
            plt.plot([epochs, epochs], plt.ylim(), label='start fine-tuning')
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend()
        if save_path is not None:
            save_path = Path(save_path)
            plt.savefig(save_path / (metric + '.png'))
        plt.close()
