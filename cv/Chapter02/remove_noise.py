#
# Created by Renatus Madrigal on 07/05/2025
#

import numpy as np
import cv2
from line_kernel import create_line_kernel
import matplotlib.pyplot as plt


def psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float("inf")  # No difference between images
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def remove_noise(
    image: np.ndarray,
    kernel_list: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """
    Remove noise from an image using morphological operations with a line kernel.

    :param image: Input image as a numpy array.
    :param kernel_list: List of line kernels to be used for morphological operations.
    :return: Image with noise removed.
    """
    opened_image = image.copy()

    for kernel in kernel_list:
        opened_image = cv2.erode(opened_image, kernel)

    factor = np.sum(np.sum(np.abs(opened_image - image)))

    return opened_image, factor


# FIXME: the noise removal process is not very ideal. Maybe something is wrong with the kernel or the method.
def main():
    kernels = [
        [create_line_kernel(5, -45), create_line_kernel(7, -45)],
        [create_line_kernel(5, 45), create_line_kernel(7, 45)],
        [create_line_kernel(3, 90), create_line_kernel(5, 90)],
        [create_line_kernel(3, 0), create_line_kernel(5, 0)],
    ]

    original_image = cv2.imread("im.jpg", cv2.IMREAD_GRAYSCALE)

    img = cv2.imread("noise.jpg", cv2.IMREAD_GRAYSCALE)
    result = []

    for kernel_set in kernels:
        result.append(remove_noise(img, kernel_set))

    factor_sum = sum(factor for _, factor in result)

    merged_image = np.zeros_like(img, dtype=np.float64)
    for opened_image, factor in result:
        merged_image += opened_image.astype(np.float64) * factor / factor_sum

    merged_image = np.clip(merged_image, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    titles = ["-45° Kernel", "45° Kernel", "90° Kernel", "0° Kernel"]
    for i, (opened_image, factor) in enumerate(result):
        axes[i + 1].imshow(opened_image, cmap="gray")
        axes[i + 1].set_title(f"{titles[i]} (Factor: {factor:.0f})")
        axes[i + 1].axis("off")

    axes[5].imshow(merged_image, cmap="gray")
    axes[5].set_title("Merged Result")
    axes[5].axis("off")

    plt.tight_layout()
    plt.show()

    # plt.clf()

    # Calculate PSNR values
    psnr_values = []
    labels = ["Noise"] + titles + ["Merged Result"]
    images = [img] + [opened_image for opened_image, _ in result] + [merged_image]

    for i, image in enumerate(images):
        psnr_val = psnr(original_image, image)
        psnr_values.append(psnr_val)

    # Create bar chart of PSNR values
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, psnr_values)
    plt.title("PSNR Comparison")
    plt.ylabel("PSNR (dB)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, psnr_values):
        if value == float("inf"):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                "∞",
                ha="center",
                va="bottom",
            )
        else:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
