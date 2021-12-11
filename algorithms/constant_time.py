import numpy as np
from tqdm import trange

from algorithms.common import MedianFilter


def get_median(kernel_hist: np.array, filter_size: int) -> int:
    """
    Find median of current window using kernel histogram

    Args:
        kernel_hist: histogram of values in current window
        filter_size: size of the filter (filter_size = 2 * radius + 1)
    Returns:
        median of current window
    """

    median = 0
    less_than_median = 0

    threshold = (filter_size ** 2) // 2

    while less_than_median + kernel_hist[median] <= threshold:
        less_than_median += kernel_hist[median]
        median += 1

    return median


class ConstantTimeMedianFilter(MedianFilter):
    def __init__(self):
        pass

    def apply(self, image: np.array, radius: int) -> np.array:
        """
        Apply Constant Time median filter of chosen radius to the image
        Based on original paper: Perreault and Hebert, "Median Filtering in Constant Time"

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
        padded_width = width + 2 * radius

        # Blurred image (filled below)
        blurred_image = np.zeros_like(image)

        for channel in trange(channels, desc="Channels"):
            # We store column hists for current row and updated ones for the next
            column_hists = np.zeros((padded_width, 256), dtype=int)
            updated_column_hists = np.zeros((padded_width, 256), dtype=int)

            # Initialize column histograms
            for j_ in range(padded_width):
                for i_ in range(filter_size):
                    value = padded_image[i_, j_, channel]

                    column_hists[j_, value] += 1
                    updated_column_hists[j_, value] += 1

            for i in range(height):
                for j in range(width):
                    if j == 0:
                        column_hists = updated_column_hists.copy()

                        # Initialize kernel histogram
                        kernel_hist = np.zeros(256, dtype=int)
                        for j_ in range(filter_size):
                            kernel_hist += column_hists[j_]

                            # Update column histograms ========================
                            if i + 1 != height:
                                # Remove old value
                                value = padded_image[i, j_, channel]
                                updated_column_hists[j_, value] -= 1

                                # Add new value
                                value = padded_image[i + filter_size, j_, channel]
                                updated_column_hists[j_, value] += 1

                    # Median is found
                    median = get_median(kernel_hist, filter_size)
                    blurred_image[i, j, channel] = median

                    # Update column histograms ================================
                    if i + 1 != height and j + 1 != width:
                        # Remove old value
                        value = padded_image[i, j + filter_size, channel]
                        updated_column_hists[j + filter_size, value] -= 1

                        # Add new value
                        value = padded_image[i + filter_size, j + filter_size, channel]
                        updated_column_hists[j + filter_size, value] += 1

                    # Update kernel histogram =================================
                    if j + 1 != width:
                        kernel_hist += column_hists[j + filter_size]
                        kernel_hist -= column_hists[j]

        return blurred_image
