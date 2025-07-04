#
# Created by Renatus Madrigal on 07/04/2025
#

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from typing import Union, Optional


def remove_fog(image: Union[str, np.ndarray], output: Optional[str] = None):
    """
    Remove fog from an image using global histogram equalization.

    @param image: Path of the input image or the image itself.
    @param output: Path to save the processed image. If None, the image will be displayed instead.
    """
    if isinstance(image, str):
        img_rgb: np.ndarray = cv2.imread(image)
        if img_rgb is None:
            raise ValueError(f"Image not found at {image}")
    elif isinstance(image, np.ndarray):
        img_rgb = image

    if img_rgb.shape[2] == 3:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    R_eq, G_eq, B_eq = map(
        lambda channel: cv2.equalizeHist(channel), cv2.split(img_rgb)
    )

    result = cv2.merge((R_eq, G_eq, B_eq))

    if output is not None:
        cv2.imwrite(output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    else:
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))

        ax[0, 0].imshow(img_rgb)
        ax[0, 0].set_title("Original Image")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(result)
        ax[0, 1].set_title("Fog Removed Image")
        ax[0, 1].axis("off")

        ax[1, 0].hist(img_rgb[:, :, 0].ravel(), bins=256, color="red", alpha=0.5)
        ax[1, 0].hist(img_rgb[:, :, 1].ravel(), bins=256, color="green", alpha=0.5)
        ax[1, 0].hist(img_rgb[:, :, 2].ravel(), bins=256, color="blue", alpha=0.5)
        ax[1, 0].set_title("Original Image Histogram")
        ax[1, 0].set_xlim([0, 255])

        ax[1, 1].hist(result[:, :, 0].ravel(), bins=256, color="red", alpha=0.5)
        ax[1, 1].hist(result[:, :, 1].ravel(), bins=256, color="green", alpha=0.5)
        ax[1, 1].hist(result[:, :, 2].ravel(), bins=256, color="blue", alpha=0.5)
        ax[1, 1].set_title("Fog Removed Image Histogram")
        ax[1, 1].set_xlim([0, 255])

        plt.tight_layout()
        plt.show()


parser = argparse.ArgumentParser(
    description="Remove fog from an image using global histogram equalization."
)
parser.add_argument("image", type=str, help="Path to the input image.")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to save the processed image. If not provided, the image will be displayed instead.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    remove_fog(args.image, args.output)
