import cv2
import numpy as np

from abc import ABC, abstractmethod


class MedianFilter(ABC):
    @abstractmethod
    def apply(self, image: np.array, radius: int) -> np.array:
        pass

    @staticmethod
    def pad_image(image: np.array, radius: int) -> np.array:
        """
        Pad image in such way to save size after filter application

        Args:
            image: image represented as numpy array
            radius: filter radius
        Returns:
            padded image represented as numpy array
        """

        return cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
