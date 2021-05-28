from abc import ABC, abstractmethod


class Explainer(ABC):
    """Abstract Base Class for COVID-19 explainers."""

    @abstractmethod
    def explain(self, image):
        """Explains the image by superimposing a heatmap.

        :param image: image to predict and explain
        :return: (prediction, confidence, explanation), where explanation is the superimposed image"""
        raise NotImplementedError
