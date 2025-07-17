import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def read_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return image
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def get_image_info(image, image_path):
    if image is None:
        return

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    print(f"Image Path: {image_path}")
    print(f"Image Dimensions: {width} x {height} pixels")
    print(f"Number of Channels: {channels}")
    print(f"Data Type: {image.dtype}")
    print(f"Image Size (pixels): {image.size}")

    file_size = os.path.getsize(image_path)
    print(f"File Size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

    if channels == 3:
        print("Color Space: BGR (Blue, Green, Red)")
    elif channels == 1:
        print("Color Space: Grayscale")
    else:
        print(f"Color Space: {channels} channels")

    print(f"Min Pixel Value: {np.min(image)}")
    print(f"Max Pixel Value: {np.max(image)}")
    print(f"Mean Pixel Value: {np.mean(image):.2f}")
    print(f"Standard Deviation: {np.std(image):.2f}")

def calculate_compression_ratio(original_image, image_path):
    if original_image is None:
        return

    height, width = original_image.shape[:2]
    channels = original_image.shape[2] if len(original_image.shape) == 3 else 1
    bytes_per_pixel = original_image.dtype.itemsize

    uncompressed_size = height * width * channels * bytes_per_pixel
    compressed_size = os.path.getsize(image_path)

    compression_ratio = uncompressed_size / compressed_size
    compression_percentage = (1 - compressed_size / uncompressed_size) * 100

    print(f"Uncompressed Size: {uncompressed_size} bytes ({uncompressed_size / (1024*1024):.2f} MB)")
    print(f"Compressed Size: {compressed_size} bytes ({compressed_size / (1024*1024):.2f} MB)")
    print(f"Compression Ratio: {compression_ratio:.2f}:1")
    print(f"Compression Percentage: {compression_percentage:.2f}%")
    print(f"Space Saved: {uncompressed_size - compressed_size} bytes")

def create_negative_image(image):
    if image is None:
        return None

    negative_image = 255 - image
    return negative_image

def display_images(original, negative, title_original="Original Image", title_negative="Negative Image"):
    if original is None or negative is None:
        return

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    negative_rgb = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title(title_original, fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(negative_rgb)
    plt.title(title_negative, fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def save_negative_image(negative_image, output_path):
    try:
        cv2.imwrite(output_path, negative_image)
        print(f"\nNegative image saved as: {output_path}")
    except Exception as e:
        print(f"Error saving negative image: {e}")

def main():
    image_path = "input.jpg"
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    print(f"\nGetting image information")
    get_image_info(original_image, image_path)

    print(f"\nCalculating compression ratio")
    calculate_compression_ratio(original_image, image_path)

    print(f"\nCreating and displaying negative image")
    negative_image = create_negative_image(original_image)

    if negative_image is not None:
        display_images(original_image, negative_image)

        negative_output_path = "negative_" + os.path.basename(image_path)
        save_negative_image(negative_image, negative_output_path)

if __name__ == "__main__":
    main()
