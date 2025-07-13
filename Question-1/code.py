import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def read_image(image_path):
    """
    Read an image using OpenCV
    """
    try:
        # Read image in color (BGR format)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return image
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def get_image_info(image, image_path):
    """
    Get comprehensive image information
    """
    if image is None:
        return

    print("=" * 50)
    print("IMAGE INFORMATION")
    print("=" * 50)

    # Basic image properties
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    print(f"Image Path: {image_path}")
    print(f"Image Dimensions: {width} x {height} pixels")
    print(f"Number of Channels: {channels}")
    print(f"Data Type: {image.dtype}")
    print(f"Image Size (pixels): {image.size}")

    # File size information
    file_size = os.path.getsize(image_path)
    print(f"File Size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

    # Color space information
    if channels == 3:
        print("Color Space: BGR (Blue, Green, Red)")
    elif channels == 1:
        print("Color Space: Grayscale")
    else:
        print(f"Color Space: {channels} channels")

    # Pixel value statistics
    print(f"Min Pixel Value: {np.min(image)}")
    print(f"Max Pixel Value: {np.max(image)}")
    print(f"Mean Pixel Value: {np.mean(image):.2f}")
    print(f"Standard Deviation: {np.std(image):.2f}")

def calculate_compression_ratio(original_image, image_path):
    """
    Calculate compression ratio by comparing uncompressed vs compressed file size
    """
    if original_image is None:
        return

    print("\n" + "=" * 50)
    print("COMPRESSION RATIO ANALYSIS")
    print("=" * 50)

    # Calculate uncompressed size (height × width × channels × bytes_per_pixel)
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
    """
    Create the negative of an image
    """
    if image is None:
        return None

    # For 8-bit images, negative is 255 - pixel_value
    negative_image = 255 - image
    return negative_image

def display_images(original, negative, title_original="Original Image", title_negative="Negative Image"):
    """
    Display original and negative images side by side
    """
    if original is None or negative is None:
        return

    # Convert BGR to RGB for matplotlib display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    negative_rgb = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)

    # Create subplot
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title(title_original, fontsize=14, fontweight='bold')
    plt.axis('off')

    # Negative image
    plt.subplot(1, 2, 2)
    plt.imshow(negative_rgb)
    plt.title(title_negative, fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def save_negative_image(negative_image, output_path):
    """
    Save the negative image to disk
    """
    try:
        cv2.imwrite(output_path, negative_image)
        print(f"\nNegative image saved as: {output_path}")
    except Exception as e:
        print(f"Error saving negative image: {e}")

def main():
    """
    Main function to execute all image processing tasks
    """
    print("DIGITAL IMAGE PROCESSING LAB - QUESTION 1")
    print("Tasks: Read image, Get info, Calculate compression ratio, Display negative")
    print("=" * 70)

    # Specify the image path (you can change this to your image file)
    image_path = input("Enter the path to your image file (or press Enter for default): ").strip()

    # Use a default path if none provided
    if not image_path:
        # Create a sample image if no path provided
        print("No image path provided. Creating a sample image...")
        sample_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        image_path = "sample_image.jpg"
        cv2.imwrite(image_path, sample_image)
        print(f"Sample image created: {image_path}")

    # Task a: Read an image
    print(f"\nTask A: Reading image from {image_path}")
    original_image = read_image(image_path)

    if original_image is None:
        print("Failed to read image. Please check the file path and try again.")
        return

    print("✓ Image successfully loaded!")

    # Task b: Get image information
    print(f"\nTask B: Getting image information")
    get_image_info(original_image, image_path)

    # Task c: Find compression ratio
    print(f"\nTask C: Calculating compression ratio")
    calculate_compression_ratio(original_image, image_path)

    # Task d: Display negative of the image
    print(f"\nTask D: Creating and displaying negative image")
    negative_image = create_negative_image(original_image)

    if negative_image is not None:
        print("✓ Negative image created successfully!")

        # Display both images
        display_images(original_image, negative_image)

        # Save negative image
        negative_output_path = "negative_" + os.path.basename(image_path)
        save_negative_image(negative_image, negative_output_path)

    print("\n" + "=" * 70)
    print("All tasks completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
