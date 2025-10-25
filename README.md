# Lab Cycle - 1

1.  Write a program to do the following tasks:
    a) Read an image.
    b) Get the image information.
    c) Find the compression ratio for the copied image.
    d) Display the negative of an image.
2.  Write a program for histogram plotting of an image.
3.  Write a to perform histogram equalization.
4.  Write a program for performing local histogram equalization.
5.  Develop programs for the following image enhancement operations.
    a) Brightness enhancement
    b) Contrast enhancement
    c) Complement of an image
    d) Bi-level or binary contrast enhancement
    e) Brightness slicing
    f) Low-pass filtering
    g) High-Pass filtering
6.  Develop programs for the following geometrical transformations on an image
    a) Translation
    b) Rotation
    c) Scaling
    d) Skewing

# Lab Cycle - 2

1.  Write a program to perform four arithmetic operations between two images.
2.  Take a noisy image. Write a program which reduces the noise by averaging the input image by 2,
    8, 16, 32 and 128. Compare all the resultant images and find which one is noise free.
3.  Write a program which implements all the types of linear spatial filters using functions.
4.  Implement a program for image convolution and correlation using a rectangular convolution
    mask of any odd size. The mask should be input as an ASCII text file. Test your program using the
    following convolution kernels:
    a) 3×3 averaging
    b) 7×7 averaging
    c) 11×11 averaging
5.  Write a program for implementing median filtering of an image. Add salt and pepper noise to it.
    Apply the median filter to the noisy image and compare the results.
6.  Detect the edges in an image using the following methods and compare the relative
    performance of these methods:
    a) Sobel
    b) Prewitt
    c) Roberts
    d) Laplacian of a Gaussian (LoG)
    e) Canny
7.  Write a program for smoothing an RGB color image with a linear spatial filter.
8.  Write a program for sharpening an RGB color image with the Laplacian filter mask.
9.  Take a sample image of size 256×256. Write a program for implementing DFT filtering with
    a) with padding and
    b) without padding, Compare the results.
10. Write a program to implement various low-pass or smoothening frequency domain filters.
11. Write a program to implement various high-pass or sharpening frequency domain filters.
12. Write a program to detect straight lines using horizontal, vertical and diagonal filter masks.

# Lab Cycle - 3

1. Write a program which restores a degraded image using direct inverse filtering.
2. Write a program for implementing Wiener filtering for linear image restoration using
   a) a constant ratio
   b) auto correlation function.
3. Write a program to convert an RGB color space to HSI. Display the Hue image, Saturation image and the Intensity image.
4. Write a program to histogram equalize the Intensity component of a color image and get a new HSI image. Convert the new HSI image back to RGB.
5. Write a program to detect the line segments in a binary image using Hough Transform.
6. Consider an image composed of small, non overlapping blobs. Write a program to segment the blobs based on thresholding.
7. Consider an image composed of small, non overlapping blobs. Write a program to segment the blobs based on region growing.
8. Write a program to implement the split and merge procedure for segmenting the image with different values for minimum dimensions of the quad-tree regions.
9. Consider a binary image composed of small blobs. Write a program to segment the blobs using watershed transform.

# Lab Cycle - 4

# Optical Flow and Block Matching Algorithm

<!-- Dataset: All experiments use the KITTI Stereo 2015 / Flow 2015 / Scene Flow 2015 dataset, using grayscale images from the image_2 folder of any sequence. -->

1. Compute dense optical flow between the frames using the Farneback algorithm and visualize the motion vectors as arrows over the first frame. Comment on differences in motion vectors in fast-moving vs slow-moving regions.

2. Select two consecutive frames from a sequence. Divide the frames into 16x16 pixel blocks. For each block in the first frame, find the best matching block in the next frame using SAD (Sum of Absolute Differences). Compute and display motion vectors as arrows over the first frame. Comment on:
   a) Which regions exhibit larger motion vectors and why?
   b) How stationary regions appear in terms of motion vectors?
   c) How the choice of block size affects the accuracy of motion estimation?
3. Select two consecutive frames from a sequence. Using motion vectors (obtained from block matching), predict the next frame from the first frame. Compute the residual difference image between the predicted frame and the actual next frame. Visualize the original frame, predicted frame, and residual image side by side. Comment on:
   a) How accurately does the predicted frame match the actual frame?
   b) Which regions have larger residual errors and why?
   c) How motion vectors contribute to reducing frame-to-frame redundancy?

# Depth Perception and Disparity Matching

<!-- Dataset: All experiments use the KITTI Stereo 2015 / Flow 2015 / Scene Flow 2015 dataset, using left and right grayscale images from the image_2 and image_3 folders of any sequence. -->

4. Select a stereo image pair (left and right images) from a KITTI sequence. Compute the disparity map using StereoSGBM. Convert the disparity map to a depth map using the camera focal length and baseline. Normalize and visualize both the disparity map and the depth map. Comment on:
   a) How disparity values relate to object distance?
   b) Which regions appear closer or farther in the depth map?

5. Select a stereo image pair (left and right images) from a KITTI sequence. Divide the left image into 16x16 pixel blocks. For each block in the left image, find the best matching block in the right image using SAD (Sum of Absolute Differences) along the same scanline. Compute and display the disparity map. Comment on:
   a) Which objects are detected as obstacles and why?
   b) How changing depth threshold affects detection results?
