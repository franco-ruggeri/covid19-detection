import tensorflow as tf
import numpy as np
import cv2
import imutils


class GradCAM:
    """Implementation of Grad-CAM using tensorflow and keras"""

    def __init__(self, model, image_path, class_index=None, layer_index=None):
        """

        :param model: The tensorflow model used
        :param image_path: The directory path to the image
        :param class_index: Most probable class label
        :param layer_index: Index of the last convolutional layer

        """

        self.model = model
        self.original_image = cv2.imread(image_path)
        self.image = None
        self.is_image_preprocessed = False
        self.class_index = class_index
        self.layer_index = layer_index

        # if layer name is not known, try to find the last convolutional layer of the model:
        if layer_index is None:
            self.layer_index = self.find_layer()

        if self.is_image_preprocessed is False:
            self.image = self.preprocess_image(image_path)
            self.is_image_preprocessed = True

        if class_index is None:
            self.class_index = self.find_class_index(self.image)

    def preprocess_image(self, image_path):
        """

        :param image_path: The directory path to the image
        :return: The preprocessed image

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

        for idx in reversed(range(len(self.model.layers))):
            output_shape = self.model.layers[idx].output.shape
            if len(output_shape) == 4:
                return idx

        raise ValueError("Could not find a matching convolutional layer, its output shape should be 4 dimensional")

    def find_class_index(self, image):
        """

        Assuming tensorflow has been used to construct the network, this method extracts the most probable label for
        the image.

        :param image: A preprocessed image, using tf.keras.preprocessing
        :return: The most likely label

        """

        predictions = self.model.predict(image)
        class_index = np.argmax(predictions[0])
        return class_index

    def generate_and_visualize_heatmap(self, output_path, alpha=0.5, eps=1e-8):

        """Generates and visualizes the heatmap used for GradCAM"""

        if self.is_image_preprocessed:

            grad_model = tf.keras.models.Model(inputs=[self.model.inputs],
                                               outputs=[self.model.get_layer(index=self.layer_index).output,
                                               self.model.output])

            with tf.GradientTape() as tape:
                inputs = tf.cast(self.image, tf.float32)
                convolutional_outputs, predictions = grad_model(inputs)
                loss = predictions[:, self.class_index]

            #output = convolutional_outputs[0]
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
            raise ValueError("The image used for generating the heatmap has not been preprocessed")


# test Grad-CAM
# if __name__ == "__main__":
#     image_path = 'data/COVIDx/train/COVID-19/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg'
#     model = tf.keras.applications.ResNet50(weights='imagenet')
#
#     original = cv2.imread(image_path)
#     resized = cv2.resize(original, (224, 224))
#
#     image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#     image = tf.keras.preprocessing.image.img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = tf.keras.applications.imagenet_utils.preprocess_input(image)
#
#     predictions = model.predict(image)
#     class_index = np.argmax(predictions[0])
#
#     decoded = tf.keras.applications.imagenet_utils.decode_predictions(predictions)
#     imagenetID, label, prob = decoded[0][0]
#     label = "{}: {:.2f}%".format(label, prob * 100)
#     print("[INFO] {}".format(label))
#
#     cam = GradCAM(model, class_index)
#     heatmap = cam.generate_heatmap(image)
#
#     heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
#     heatmap, output = cam.overlay_heatmap(heatmap, original, alpha=0.5)
#
#     cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
#     cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8, (255, 255, 255), 2)
#     # display the original image and resulting heatmap and output image
#     # to our screen
#     output = np.vstack([original, heatmap, output])
#     output = imutils.resize(output, height=700)
#     cv2.imshow("Output", output)
#     cv2.waitKey(0)
