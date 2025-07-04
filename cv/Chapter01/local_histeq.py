#
# Created by Renatus Madrigal on 07/04/2025
#

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
from typing import Union, Optional


def remove_fog_by_local_histeq(
    image_path: Union[str, np.ndarray], output: Optional[str] = None
):
    # Read image in BGR format and convert to RGB
    if isinstance(image_path, str):
        img_bgr: np.ndarray = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Image not found at {image_path}")
    elif isinstance(image_path, np.ndarray):
        img_bgr = image_path

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(img_rgb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    eq_R = clahe.apply(R)
    eq_G = clahe.apply(G)
    eq_B = clahe.apply(B)

    img_eq = cv2.merge((eq_R, eq_G, eq_B))

    if output is not None:
        # Save the processed image
        cv2.imwrite(output, cv2.cvtColor(img_eq, cv2.COLOR_RGB2BGR))
    else:
        # Convert to grayscale for histograms
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(img_eq)
        axs[0, 1].set_title("CLAHE Enhanced")
        axs[0, 1].axis("off")

        axs[1, 0].hist(img_rgb[:, :, 0].ravel(), bins=256, color="red", alpha=0.5)
        axs[1, 0].hist(img_rgb[:, :, 1].ravel(), bins=256, color="green", alpha=0.5)
        axs[1, 0].hist(img_rgb[:, :, 2].ravel(), bins=256, color="blue", alpha=0.5)
        axs[1, 0].set_title("Original RGB Channel Histogram")
        axs[1, 0].set_xlim([0, 255])

        axs[1, 1].hist(img_eq[:, :, 0].ravel(), bins=256, color="red", alpha=0.5)
        axs[1, 1].hist(img_eq[:, :, 1].ravel(), bins=256, color="green", alpha=0.5)
        axs[1, 1].hist(img_eq[:, :, 2].ravel(), bins=256, color="blue", alpha=0.5)
        axs[1, 1].set_title("CLAHE Enhanced Histogram")

        plt.tight_layout()
        plt.show()


parser = argparse.ArgumentParser(
    description="Remove fog from an image using local histogram equalization."
)
parser.add_argument("image_path", type=str, help="Path to the input image.")
parser.add_argument(
    "--output", type=str, default=None, help="Path to save the processed image."
)


if __name__ == "__main__":
    args = parser.parse_args()
    remove_fog_by_local_histeq(args.image_path, args.output)
