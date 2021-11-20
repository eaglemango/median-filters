import numpy as np
from tqdm import trange

from .common import MedianFilter


class HuangMedianFilter(MedianFilter):
    def __init__(self):
        pass

    def apply(self, image: np.array, radius: int) -> np.array:
        """
        Apply Huang median filter of chosen radius to the image
        Based on original paper: Huang et al., "A Fast Two-Dimensional Median Filtering Algorithm"

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

        # Threshold for median identification
        threshold = (filter_size ** 2) // 2

        for i in trange(height):
            for channel in range(channels):
                # Build new histogram for each row
                hist = np.zeros(256)

                # Values to be updated during iteration over row
                median = 0
                less_than_median = 0

                for j in range(width - 1):
                    # Starting window
                    if j == 0:
                        # Fill initial histogram
                        for i_ in range(i, i + filter_size):
                            for j_ in range(j, j + filter_size):
                                hist[padded_image[i_, j_, channel]] += 1

                        # Find starting median
                        while less_than_median + hist[median] <= threshold:
                            less_than_median += hist[median]
                            median += 1

                        blurred_image[i, j, channel] = median

                    # Move to the next window in current row
                    prev_left_column = padded_image[i:i+filter_size, j, channel]
                    next_right_column = padded_image[i:i+filter_size, j+filter_size, channel]

                    # Update histogram
                    for i_ in range(filter_size):
                        # Remove old values ========
                        value = prev_left_column[i_]

                        hist[value] -= 1
                        if value < median:
                            less_than_median -= 1

                        # Add new values ============
                        value = next_right_column[i_]

                        hist[value] += 1
                        if value < median:
                            less_than_median += 1

                    # Update median if needed
                    if less_than_median > threshold:
                        median -= 1
                        less_than_median -= hist[median]

                        while less_than_median > threshold:
                            median -= 1
                            less_than_median -= hist[median]
                    else:
                        while less_than_median + hist[median] <= threshold:
                            less_than_median += hist[median]
                            median += 1

                    # Median is found
                    blurred_image[i, j + 1, channel] = median

        return blurred_image
