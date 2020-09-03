import numpy as np
import tensorflow as tf
from scipy import ndimage
 

def _apply_polarity(attributions, polarity):
    if polarity == "positive":
        return np.clip(attributions, 0, 1)
    else:
        return np.clip(attributions, -1, 0)


def _get_thresholded_attributions(attributions, percentage):
    if percentage == 100.0:
        return np.min(attributions)

    # 1. Flatten the attributions
    flatten_attr = attributions.flatten()

    # 2. Get the sum of the attributions
    total = np.sum(flatten_attr)

    # 3. Sort the attributions from largest to smallest.
    sorted_attributions = np.sort(np.abs(flatten_attr))[::-1]

    # 4. Calculate the percentage of the total sum that each attribution
    # and the values about it contribute.
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / total

    # 5. Threshold the attributions by the percentage
    indices_to_consider = np.where(cum_sum >= percentage)[0][0]

    # 6. Select the desired attributions and return
    attributions = sorted_attributions[indices_to_consider]
    return attributions


def binarize(attributions, threshold=0.001):
    return attributions > threshold


def _apply_linear_transformation(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0,
                                 lower_end=0.2):
    # 1. Get the thresholds
    m = _get_thresholded_attributions(
        attributions, percentage=100 - clip_above_percentile
    )
    e = _get_thresholded_attributions(
        attributions, percentage=100 - clip_below_percentile
    )

    # 2. Transform the attributions by a linear function f(x) = a*x + b such that
    # f(m) = 1.0 and f(e) = lower_end
    transformed_attributions = (1 - lower_end) * (np.abs(attributions) - e) / (
        m - e
    ) + lower_end

    # 3. Make sure that the sign of transformed attributions is the same as original attributions
    transformed_attributions *= np.sign(attributions)

    # 4. Only keep values that are bigger than the lower_end
    transformed_attributions *= transformed_attributions >= lower_end

    # 5. Clip values and return
    transformed_attributions = np.clip(transformed_attributions, 0.0, 1.0)
    return transformed_attributions


def _morphological_cleanup_fn(attributions, structure=np.ones((4, 4))):
    closed = ndimage.grey_closing(attributions, structure=structure)
    opened = ndimage.grey_opening(closed, structure=structure)
    return opened


def _draw_outlines(attributions, percentage=90, connected_component_structure=np.ones((3, 3))):
    # 1. Binarize the attributions.
    attributions = binarize(attributions)

    # 2. Fill the gaps
    attributions = ndimage.binary_fill_holes(attributions)

    # 3. Compute connected components
    connected_components, num_comp = ndimage.measurements.label(
        attributions, structure=connected_component_structure
    )

    # 4. Sum up the attributions for each component
    total = np.sum(attributions[connected_components > 0])
    component_sums = []
    for comp in range(1, num_comp + 1):
        mask = connected_components == comp
        component_sum = np.sum(attributions[mask])
        component_sums.append((component_sum, mask))

    # 5. Compute the percentage of top components to keep
    sorted_sums_and_masks = sorted(component_sums, key=lambda x: x[0], reverse=True)
    sorted_sums = list(zip(*sorted_sums_and_masks))[0]
    cumulative_sorted_sums = np.cumsum(sorted_sums)
    cutoff_threshold = percentage * total / 100
    cutoff_idx = np.where(cumulative_sorted_sums >= cutoff_threshold)[0][0]
    if cutoff_idx > 2:
        cutoff_idx = 2

    # 6. Set the values for the kept components
    border_mask = np.zeros_like(attributions)
    for i in range(cutoff_idx + 1):
        border_mask[sorted_sums_and_masks[i][1]] = 1

    # 7. Make the mask hollow and show only the border
    eroded_mask = ndimage.binary_erosion(border_mask, iterations=1)
    border_mask[eroded_mask] = 0

    # 8. Return the outlined mask
    return border_mask


class IG:
    """
    IG (Integrated Gradients).

    Supported models: sub-package covid19.models
    Adapted from: TODO
    """

    def __init__(self, model, positive_channel=None, negative_channel=None):
        self.model = model
        self.image_shape = model.image_shape

        if positive_channel is None:
            self.positive_channel = [0, 255, 0]
        else:
            self.positive_channel = positive_channel

        if negative_channel is None:
            self.negative_channel = [255, 0, 0]
        else:
            self.negative_channel = negative_channel

    def _get_gradients(self, img_input, top_pred_idx):
        """Computes the gradients of outputs w.r.t input image.

        Args:
            img_input: 4D image tensor
            top_pred_idx: Predicted label for the input image

        Returns:
            Gradients of the predictions w.r.t img_input
        """
        images = tf.cast(img_input, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(images)
            preds = self.model(images)
            top_class = preds[:, top_pred_idx]

        grads = tape.gradient(top_class, images)
        return grads

    def _get_integrated_gradients(self, img_input, top_pred_idx, baseline=None, num_steps=50):
        """Computes Integrated Gradients for a predicted label.

        Args:
            img_input (ndarray): Original image
            top_pred_idx: Predicted label for the input image
            baseline (ndarray): The baseline image to start with for interpolation
            num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

            Returns:
            Integrated gradients w.r.t input image
        """
        # If baseline is not provided, start with a black image
        # having same size as the input image.
        if baseline is None:
            baseline = np.zeros(self.image_shape).astype(np.float32)
        else:
            baseline = baseline.astype(np.float32)

        # 1. Do interpolation.
        img_input = img_input.astype(np.float32)
        interpolated_image = [
            baseline + (step / num_steps) * (img_input - baseline)
            for step in range(num_steps + 1)
        ]
        interpolated_image = np.array(interpolated_image).astype(np.float32)

        # 2. Preprocess the interpolated images
        interpolated_image = self.model.preprocess(interpolated_image)

        # 3. Get the gradients
        grads = []
        for i, img in enumerate(interpolated_image):
            img = tf.expand_dims(img, axis=0)
            grad = self._get_gradients(img, top_pred_idx=top_pred_idx)
            grads.append(grad[0])
        grads = tf.convert_to_tensor(grads, dtype=tf.float32)

        # 4. Approximate the integral using the trapezoidal rule
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)

        # 5. Calculate integrated gradients and return
        integrated_grads = (img_input - baseline) * avg_grads
        return integrated_grads

    def _random_baseline_integrated_gradients(self, img_input, top_pred_idx, num_steps=50, num_runs=2):
        """Generates a number of random baseline images.

        Args:
            img_input (ndarray): 3D image
            top_pred_idx: Predicted label for the input image
            num_steps: Number of interpolation steps between the baseline
                and the input used in the computation of integrated gradients. These
                steps along determine the integral approximation error. By default,
                num_steps is set to 50.
            num_runs: number of baseline images to generate

        Returns:
            Averaged integrated gradients for `num_runs` baseline images
        """
        # 1. List to keep track of Integrated Gradients (IG) for all the images
        integrated_grads = []

        # 2. Get the integrated gradients for all the baselines
        for run in range(num_runs):
            baseline = np.random.random(self.image_shape) * 255
            igrads = self._get_integrated_gradients(img_input=img_input, top_pred_idx=top_pred_idx, baseline=baseline,
                                                    num_steps=num_steps)
            integrated_grads.append(igrads)

        # 3. Return the average integrated gradients for the image
        integrated_grads = tf.convert_to_tensor(integrated_grads)
        return tf.reduce_mean(integrated_grads, axis=0)

    def _process_grads(self, image, attributions, polarity="positive", clip_above_percentile=99.9,
                       clip_below_percentile=0, morphological_cleanup=False, structure=np.ones((3, 3)),
                       outlines=False, outlines_component_percentage=90, overlay=True):
        if polarity not in ["positive", "negative"]:
            raise ValueError(
                f""" Allowed polarity values: 'positive' or 'negative'
                                but provided {polarity}"""
            )
        if clip_above_percentile < 0 or clip_above_percentile > 100:
            raise ValueError("clip_above_percentile must be in [0, 100]")

        if clip_below_percentile < 0 or clip_below_percentile > 100:
            raise ValueError("clip_below_percentile must be in [0, 100]")

        # 1. Apply polarity
        if polarity == "positive":
            attributions = _apply_polarity(attributions, polarity=polarity)
            channel = self.positive_channel
        else:
            attributions = _apply_polarity(attributions, polarity=polarity)
            attributions = np.abs(attributions)
            channel = self.negative_channel

        # 2. Take average over the channels
        attributions = np.average(attributions, axis=2)

        # 3. Apply linear transformation to the attributions
        attributions = _apply_linear_transformation(
            attributions,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            lower_end=0.0,
        )

        # 4. Cleanup
        if morphological_cleanup:
            attributions = _morphological_cleanup_fn(
                attributions, structure=structure
            )
        # 5. Draw the outlines
        if outlines:
            attributions = _draw_outlines(
                attributions, percentage=outlines_component_percentage
            )

        # 6. Expand the channel axis and convert to RGB
        attributions = np.expand_dims(attributions, 2) * channel

        # 7.Superimpose on the original image
        if overlay:
            attributions = np.clip((attributions * 0.8 + image), 0, 255)
        return attributions

    def _visualize(self, image, integrated_gradients, polarity="positive", clip_above_percentile=99.9,
                   clip_below_percentile=0, morphological_cleanup=False, structure=np.ones((3, 3)), outlines=False,
                   outlines_component_percentage=90, overlay=True, figsize=(15, 8)):    # TODO: figsize not used
        # 1. Make two copies of the original image
        img2 = np.copy(image)

        # 3. Process the integrated gradients
        igrads_attr = self._process_grads(
            image=img2,
            attributions=integrated_gradients,
            polarity=polarity,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            morphological_cleanup=morphological_cleanup,
            structure=structure,
            outlines=outlines,
            outlines_component_percentage=outlines_component_percentage,
            overlay=overlay,
        )

        return igrads_attr
        # return igrads_attr.astype(np.unit8) if the first doesnt work

    def explain(self, image):
        # 2. Keep a copy of the original image
        orig_img = np.copy(image).astype(np.uint8)

        # 3. Preprocess the image
        img_processed = tf.cast(self.model.preprocess(image), dtype=tf.float32)

        # 4. Get model predictions
        preds = self.model.predict(img_processed)
        top_pred_idx = tf.argmax(preds[0])

        # 5. Get the gradients of the last layer for the predicted label
        # TODO: ???? comment above and no instructions?
        # Milan : The instruction was for the normal gradients as I recall, I just deleted the line (considering we dont need those pictures) without deleting the comment from tutorial, my error;

        # 6. Get the integrated gradients
        igrads = self._random_baseline_integrated_gradients(np.copy(orig_img), top_pred_idx=top_pred_idx, num_steps=50,
                                                            num_runs=2)

        explanation = self._visualize(image=orig_img, integrated_gradients=igrads.numpy(), clip_above_percentile=95,
                                      clip_below_percentile=28, morphological_cleanup=True, outlines=True)

        return top_pred_idx, explanation
