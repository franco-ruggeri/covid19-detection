import matplotlib.pyplot as plt


def plot_explanation(original, explanation, prediction, ground_truth, save_path=None):
    """
    Plots the original image alongside the explanation.

    :param original: numpy array of shape (height, width, channels), original image
    :param explanation: numpy array of shape (height, width, channels), heatmap superimposed to original image
    :param prediction: string, name of predicted class
    :param ground_truth: string, name of true class
    :param save_path: path where to save the figure
    :return:
    """
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(original / 255)
    plt.axis('off')
    plt.title('original')

    plt.subplot(1, 2, 2)
    plt.imshow(explanation)
    plt.axis('off')
    plt.title('explanation')
    plt.suptitle('Prediction: {}\nGround truth: {}'.format(prediction, ground_truth))

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
