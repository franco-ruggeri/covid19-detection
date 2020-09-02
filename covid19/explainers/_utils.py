import matplotlib.pyplot as plt


def plot_explanation(original_image, explanation, title=None, save_path=None):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(original_image / 255)
    plt.subplot(1, 2, 2)
    plt.imshow(explanation)
    if title is not None:
        plt.suptitle(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
