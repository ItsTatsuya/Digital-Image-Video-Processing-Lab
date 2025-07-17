import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('input.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=color, label=color.capitalize())

plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('RGB Histogram')
plt.legend()

plt.tight_layout()
plt.show()
