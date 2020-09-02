import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping).

    Supported models: sub-package covid19.models
    Adapted from: https://keras.io/examples/vision/grad_cam/
    """

    def __init__(self, model):
        # model mapping input image to activations of last conv layer
        inputs = Input(shape=model.image_shape)
        x = model.preprocess(inputs)
        outputs = model.feature_extractor(x)
        self.conv_base = Model(inputs=inputs, outputs=outputs)

        # model mapping activations of last conv layer to predictions
        inputs = Input(shape=outputs.shape[1:])
        x = inputs
        for layer in model.classifier:
            x = layer(x)
        outputs = x
        self.classifier = Model(inputs=inputs, outputs=outputs)

    def _make_heatmap(self, image):
        image = tf.expand_dims(image, axis=0)   # add batch dimension

        with tf.GradientTape() as tape:
            last_conv_activations = self.conv_base(image)
            tape.watch(last_conv_activations)
            probabilities = self.classifier(last_conv_activations)
            prediction = tf.argmax(probabilities[0])
            top_probability = probabilities[:, prediction]

        grads = tape.gradient(top_probability, last_conv_activations)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_activations = last_conv_activations.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_activations[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(last_conv_activations, axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)      # range [0, 1]
        return heatmap

    def explain(self, image, save_path=None):
        # compute heatmap
        heatmap = self._make_heatmap(image)                     # range [0, 1]
        heatmap = np.uint8(255 * heatmap)                       # range [0, 255] (for colormap)

        # colorize new image according to heatmap (with colormap 'jet')
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
        jet_heatmap = img_to_array(jet_heatmap)

        # superimposed image
        superimposed_image = jet_heatmap * 0.4 + image
        superimposed_image /= np.max(superimposed_image)        # range [0, 1] (for plt.imshow())

        # save and show
        plt.figure()
        plt.imshow(superimposed_image)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()