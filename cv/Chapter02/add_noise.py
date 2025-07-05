#
# Created by Renatus Madrigal on 07/05/2025
#

import cv2
import numpy as np

# Read the image
img = cv2.imread("im.jpg", cv2.IMREAD_GRAYSCALE)

# Generate Poisson noise
# Scale image to appropriate range for Poisson distribution
factor = 0.5
scaled = img.astype(np.float64) * factor
noisy = np.random.poisson(scaled) / factor

# Convert back to uint8 and clip values
noisy_img = np.clip(noisy, 0, 255).astype(np.uint8)

# Save the noisy image
cv2.imwrite("noise.jpg", noisy_img)
