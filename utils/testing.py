import numpy as np
from time import time
from typing import List

from algorithms.common import MedianFilter


def measure_algorithm_speed(algorithm: MedianFilter, image: np.array, radius: int) -> float:
    """
    Measure speed of chosen algorithm in msecs/megapixels

    Args:
        algorithm: median filtering algorithm
        image: image represented as numpy array
        radius: filter radius
    Returns:
        algorithm speed in msecs/megapixels
    """

    start = time()
    algorithm.apply(image, radius)
    finish = time()

    # Time in milliseconds
    total_time = (finish - start) * 1000

    # Image size in megapixels
    height, width, channels = image.shape
    image_size = height * width / 1000000

    # Milliseconds per megapixel
    speed = total_time / image_size

    return speed


def test_algorithms(algorithms: List[MedianFilter], image: np.array, radii: List[int]):
    measurements = {}

    for algorithm in algorithms:
        algorithm_name = algorithm.__name__
        algorithm_instance = algorithm()
        print(f"=== Testing {algorithm_name} ===")

        for radius in radii:
            print(f"Radius = {radius}")

            speed = measure_algorithm_speed(algorithm_instance, image, radius)

            if algorithm_name not in measurements:
                measurements[algorithm_name] = []

            measurements[algorithm_name].append(speed)

            print(f"Algorithm Speed = {speed:.2f} msecs/megapixels\n")

    return measurements
