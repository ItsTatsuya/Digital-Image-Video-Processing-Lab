import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('input.jpg', 0)

hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])

equalized_img = cv2.equalizeHist(img)

hist_equalized = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.plot(hist_original)
plt.title('Original Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(2, 3, 3)
cdf = hist_original.cumsum()
cdf_normalized = cdf * hist_original.max() / cdf.max()
plt.plot(cdf_normalized, color='b')
plt.title('Cumulative Distribution Function')
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Frequency')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.imshow(equalized_img, cmap='gray')
plt.title('Histogram Equalized Image')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.plot(hist_equalized)
plt.title('Equalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(2, 3, 6)
cdf_eq = hist_equalized.cumsum()
cdf_eq_normalized = cdf_eq * hist_equalized.max() / cdf_eq.max()
plt.plot(cdf_eq_normalized, color='r')
plt.title('Equalized CDF')
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()

cv2.imwrite('equalized_output.jpg', equalized_img)
