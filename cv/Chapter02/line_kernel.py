#
# Created by Renatus Madrigal on 07/05/2025
#

import numpy as np
import math


def create_line_kernel(length, angle_deg) -> np.ndarray:
    """
    Create a line kernel for morphological operations, equivalent to MATLAB's `strel.line`.

    :param length: Length of the line in pixels. If even, it will be automatically converted to odd.
    :param angle_deg: Angle of the line in degrees, must be a multiple of 45.
    """
    # FIXME: Currently this function only works for angles that are multiples of 45 degrees.

    if length % 2 == 0:
        length += 1  # Automatically convert to odd

    # Calculate direction vector
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = -math.sin(angle_rad)

    # Calculate bounding box size (MATLAB algorithm)
    # Reference: https://www.mathworks.com/help/images/ref/strel.line.html
    sz = 2 * np.floor((length - 1) / 2 * abs(np.array([dy, dx]))).astype(int) + 1
    kernel = np.zeros(sz, dtype=np.uint8)

    # Calculate center point
    center = (np.array(kernel.shape) - 1) // 2
    center = np.array([center[1], center[0]])  # Adjust for row-major order

    # Calculate endpoint coordinates
    half_len = (length - 1) / 2
    end1 = center - np.round(half_len * np.array([dx, dy])).astype(int)
    end2 = center + np.round(half_len * np.array([dx, dy])).astype(int)

    # Bresenham line algorithm (exact match to MATLAB)
    points = []
    x0, y0 = end1
    x1, y1 = end2

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    # print(f"Center: {center}")  # Debugging output
    # print(f"End points: {end1}, {end2}")  # Debugging output
    # print(f"Direction vector: ({dx}, {dy})")  # Debugging output
    # print(f"Kernel shape: {kernel.shape}")  # Debugging output
    # print(f"Line points: {points}")  # Debugging output

    # Set kernel values to 1 at the calculated points
    for y, x in points:
        kernel[x, y] = 1

    return kernel


if __name__ == "__main__":
    print(create_line_kernel(5, 0))
    # print(create_line_kernel(5, 30))
    print(create_line_kernel(5, 45))
    print(create_line_kernel(5, 90))
    print(create_line_kernel(5, -45))
    print(create_line_kernel(7, -45))
