#
# Created by Renatus Madrigal on 07/04/2025
#

import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Union, Optional
import cv2


def get_gaussian_kernel(sigma: float, kernel_size: int) -> np.ndarray:
    """
    Generate a Gaussian kernel.

    @param sigma: Standard deviation of the Gaussian.
    @param kernel_size: Size of the kernel (must be odd).
    @return: Gaussian kernel as a 2D numpy array.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


def retinex_impl(
    img_rgb: np.ndarray,
    sigmas: tuple[float, float, float] = (60, 30, 15),
    use_opencv_gaussian: bool = True,
) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    min_dim = min(h, w)
    eps = 1e-10

    image_float = img_rgb.astype(np.float32) / 255.0
    r, g, b = cv2.split(image_float)

    r_refl, g_refl, b_refl = map(np.zeros_like, (r, g, b))

    for sigma in sigmas:
        kernel_size = int(6 * sigma)
        kernel_size = min(kernel_size, min_dim)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

        if use_opencv_gaussian:
            r_blur, g_blur, b_blur = map(
                lambda channel: cv2.GaussianBlur(
                    channel, (kernel_size, kernel_size), sigma
                ),
                (r, g, b),
            )
        else:
            kernel = get_gaussian_kernel(sigma, kernel_size)
            r_blur, g_blur, b_blur = map(
                lambda channel: cv2.filter2D(channel, -1, kernel),
                (r, g, b),
            )

        scale_weight = 1.0 / len(sigmas)  # Average the results from different sigmas

        r_refl += scale_weight * (np.log(r + eps) - np.log(r_blur + eps))
        g_refl += scale_weight * (np.log(g + eps) - np.log(g_blur + eps))
        b_refl += scale_weight * (np.log(b + eps) - np.log(b_blur + eps))

    def normalize_channel(channel: np.ndarray) -> np.ndarray:
        # Normalize the channel to range [0, 255]
        channel = (channel - np.min(channel)) / (
            np.max(channel) - np.min(channel) + eps
        )
        channel = (channel * 255).astype(np.uint8)
        return channel

    img_retinex = cv2.merge(
        (
            normalize_channel(r_refl),
            normalize_channel(g_refl),
            normalize_channel(b_refl),
        )
    )

    return img_retinex


def remove_fog_by_retinex(image: Union[str, np.ndarray], output: Optional[str] = None):
    """
    Remove fog from an image using Retinex-based method.

    @param image: Path of the input image or the image itself.
    @param output: Path to save the processed image. If None, the image will be displayed instead.
    """
    if isinstance(image, str):
        img_bgr: np.ndarray = cv2.imread(image)
        if img_bgr is None:
            raise ValueError(f"Image not found at {image}")
    elif isinstance(image, np.ndarray):
        img_bgr = image

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_retinex = retinex_impl(img_rgb, use_opencv_gaussian=False)

    if output is not None:
        cv2.imwrite(output, cv2.cvtColor(img_retinex, cv2.COLOR_RGB2BGR))
    else:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        axs[0, 0].imshow(img_rgb)
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(img_retinex)
        axs[0, 1].set_title("Retinex Enhanced")
        axs[0, 1].axis("off")

        axs[1, 0].hist(img_rgb[:, :, 0].ravel(), bins=256, color="red", alpha=0.5)
        axs[1, 0].hist(img_rgb[:, :, 1].ravel(), bins=256, color="green", alpha=0.5)
        axs[1, 0].hist(img_rgb[:, :, 2].ravel(), bins=256, color="blue", alpha=0.5)
        axs[1, 0].set_title("Original RGB Channel Histogram")
        axs[1, 0].set_xlim([0, 255])

        axs[1, 1].hist(img_retinex[:, :, 0].ravel(), bins=256, color="red", alpha=0.5)
        axs[1, 1].hist(img_retinex[:, :, 1].ravel(), bins=256, color="green", alpha=0.5)
        axs[1, 1].hist(img_retinex[:, :, 2].ravel(), bins=256, color="blue", alpha=0.5)
        axs[1, 1].set_title("Retinex Enhanced Histogram")

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
    remove_fog_by_retinex(args.image_path, args.output)
