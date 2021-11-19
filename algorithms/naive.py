import numpy as np
from tqdm import trange

from .common import MedianFilter


class NaiveMedianFilter(MedianFilter):
    def __init__(self):
        pass

    def apply(self, image: np.array, radius: int) -> np.array:
        """
        Apply "naive" median filter of chosen radius to the image
        "Naive" means using sorting for finding the median value

        Args:
            image: image represented as numpy array
            radius: filter radius
        Returns:
            image after filter application
        """

        # Array dimensions
        height, width, channels = image.shape
        filter_size = 2 * radius + 1

        # We need to pad image before using filter
        padded_image = self.pad_image(image, radius)

        # Blurred image (filled below)
        blurred_image = np.zeros_like(image)

        # Iterate over all pixels and find median in each filter window
        for i in trange(height):
            for j in range(width):
                image_slice = padded_image[i:i+filter_size, j:j+filter_size, :].transpose((2, 0, 1))

                pixels_count = filter_size ** 2

                image_slice = image_slice.reshape((channels, pixels_count))
                medians = np.sort(image_slice, axis=-1)[:, pixels_count // 2]

                blurred_image[i][j] = medians

        return blurred_image
