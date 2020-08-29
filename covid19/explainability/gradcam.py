import tensorflow as tf
import numpy as np
import cv2
import imutils

class GradCAM:
    """Implementation of Grad-CAM using tensorflow and keras"""

    def __init__(self, model, image_path, class_index=None, layer_name=None, composed_model=False):
        """

        :param model: The tensorflow model used
        :param image_path: The directory path to the preprocessing
        :param class_index: Most probable class label
        :param layer_index: Index of the last convolutional layer

        """
        self.composed_model = composed_model
        self.model = model
        self.original_image = cv2.imread(image_path)
        self.image = None
        self.is_image_preprocessed = False
        self.class_index = class_index
        self.layer_name = layer_name

        # if layer name is not known, try to find the last convolutional layer of the model:
        if layer_name is None:
            self.layer_name = self.find_layer()

        if class_index is None:
            self.class_index = self.find_class_index(self.image)

    def preprocess_image(self, image_path):
        """

        :param image_path: The directory path to the preprocessing
        :return: The preprocessed preprocessing

        """

        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.imagenet_utils.preprocess_input(image)

        return image

    def find_layer(self):
        """

        :param model: The model of the network, assumed to be built with tensorflow.
        :return: The last convolutional layer of the network, to be used for the guided backpropagation gradient.

        """

        # iterate over the layers in reversed order, when a four dimensional output is encountered this is assumed to be
        # the convolutional layer

        if self.composed_model:

            for layer in reversed(self.model.layers):
                if layer.name == 'resnet50':

                    for model_layer in reversed(layer.layers):

                        output_shape = model_layer.output.shape
                        if len(output_shape) == 4:
                            return model_layer.name
        else:

            for idx in reversed(range(len(self.model.layers))):
                output_shape = self.model.layers[idx].output.shape
                if len(output_shape) == 4:
                    return idx

        raise ValueError("Could not find a matching convolutional layer, its output shape should be 4 dimensional")

    def find_class_index(self, image):
        """

        Assuming tensorflow has been used to construct the network, this method extracts the most probable label for
        the preprocessing.

        :param image: A preprocessed preprocessing, using tf.keras.preprocessing
        :return: The most likely label

        """

        predictions = self.model.predict(image)
        print(predictions)
        class_index = np.argmax(predictions[0])
        return class_index

    def generate_and_visualize_heatmap(self, output_path, alpha=0.5, eps=1e-8):

        """Generates and visualizes the heatmap used for GradCAM"""

        if self.is_image_preprocessed:

            if self.composed_model:

                inputs = [self.model.get_layer('resnet50').inputs]
                outputs = [self.model.get_layer('resnet50').get_layer(self.layer_name).output,
                           self.model.get_layer('resnet50').output]

            else:

                inputs = [self.model.inputs]
                outputs = [self.model.get_layer(self.layer_name).output, self.model.output]

            grad_model = tf.keras.models.Model(inputs=inputs,
                                               outputs=outputs)

            with tf.GradientTape() as tape:
                inputs = tf.cast(self.image, tf.float32)
                convolutional_outputs, predictions = grad_model(inputs)
                loss = predictions[:, self.class_index]

            # output = convolutional_outputs[0]
            gradients = tape.gradient(loss, convolutional_outputs)

            cast_conv_outputs = tf.cast(convolutional_outputs > 0, "float32")
            cast_gradients = tf.cast(gradients > 0, "float32")
            guided_gradients = cast_conv_outputs * cast_gradients * gradients

            convolutional_outputs = convolutional_outputs[0]
            guided_gradients = guided_gradients[0]

            weights = tf.reduce_mean(guided_gradients, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, convolutional_outputs), axis=-1)

            width, height = self.image.shape[2], self.image.shape[1]
            heatmap = cv2.resize(cam.numpy(), (width, height))

            numer = heatmap - np.min(heatmap)
            denom = (heatmap.max() - heatmap.min()) + eps
            heatmap = numer / denom
            heatmap = (heatmap * 255).astype("uint8")

            heatmap = cv2.resize(heatmap, (self.original_image.shape[1], self.original_image.shape[0]))
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            output = cv2.addWeighted(self.original_image, alpha, heatmap, 1 - alpha, 0)

            display_output = np.vstack([self.original_image, heatmap, output])
            display_output = imutils.resize(display_output, height=700)
            # cv2.imshow("output", display_output)
            cv2.imwrite(output_path, display_output)
            cv2.waitKey(0)

        else:
            raise ValueError("The preprocessing used for generating the heatmap has not been preprocessed")
