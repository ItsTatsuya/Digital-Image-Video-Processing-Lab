
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib for better visualization
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.size'] = 10

# Define paths (relative to current script location)
script_dir = Path(__file__).parent
dataset_path = script_dir.parent / 'dataset'
image_folder = dataset_path / 'training' / 'image_0'
output_folder = script_dir / 'output'
output_folder.mkdir(exist_ok=True)

# Get list of image files (sorted)
image_files = sorted(list(image_folder.glob('*.png')))

# Select a sequence with good motion (sequence 10 has good car movement)
sequence_idx = 10
frame1_path = image_folder / f'{sequence_idx:06d}_10.png'
frame2_path = image_folder / f'{sequence_idx:06d}_11.png'

# Load consecutive frames in grayscale
frame1 = cv2.imread(str(frame1_path), cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread(str(frame2_path), cv2.IMREAD_GRAYSCALE)

print(f"Frame 1: {frame1_path.name}")
print(f"Frame 2: {frame2_path.name}")
print(f"Frame shape: {frame1.shape}")
print(f"Frame dtype: {frame1.dtype}")

# Display the frames
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
axes[0].imshow(frame1, cmap='gray')
axes[0].set_title('Frame 1 (t)', fontsize=14)
axes[0].axis('off')

axes[1].imshow(frame2, cmap='gray')
axes[1].set_title('Frame 2 (t+1)', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig(output_folder / '01_input_frames.png', dpi=150, bbox_inches='tight')
print(f"Saved: 01_input_frames.png")
plt.show()

# Compute dense optical flow using Farneback algorithm
# Parameters:
# - pyr_scale: image pyramid scale (< 1) - 0.5 means each layer is half the size
# - levels: number of pyramid layers
# - winsize: averaging window size for gaussian blur
# - iterations: number of iterations at each pyramid level
# - poly_n: size of pixel neighborhood for polynomial expansion (5 or 7)
# - poly_sigma: standard deviation of Gaussian for polynomial expansion
# - flags: operation flags

flow = cv2.calcOpticalFlowFarneback(
    frame1, frame2,
    None,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

print(f"Flow shape: {flow.shape}")
print(f"Flow dtype: {flow.dtype}")
print(f"Flow range - X: [{flow[..., 0].min():.2f}, {flow[..., 0].max():.2f}]")
print(f"Flow range - Y: [{flow[..., 1].min():.2f}, {flow[..., 1].max():.2f}]")

# Calculate magnitude and angle of flow vectors
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

print(f"\nMagnitude range: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
print(f"Mean magnitude: {magnitude.mean():.2f}")

# Create HSV representation of optical flow
hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
hsv[..., 1] = 255  # Full saturation

# Hue represents direction, value represents magnitude
hsv[..., 0] = angle * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

# Convert HSV to RGB for visualization
flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Display the flow visualization
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

axes[0].imshow(frame1, cmap='gray')
axes[0].set_title('Frame 1', fontsize=14)
axes[0].axis('off')

axes[1].imshow(magnitude, cmap='hot')
axes[1].set_title('Flow Magnitude', fontsize=14)
axes[1].axis('off')
plt.colorbar(axes[1].imshow(magnitude, cmap='hot'), ax=axes[1], fraction=0.046)

axes[2].imshow(flow_rgb)
axes[2].set_title('Flow Direction (HSV)', fontsize=14)
axes[2].axis('off')

plt.tight_layout()
plt.savefig(output_folder / '02_flow_visualization.png', dpi=150, bbox_inches='tight')
print(f"Saved: 02_flow_visualization.png")
plt.show()

print("HSV Flow Legend:")
print("- Hue (color) represents direction of motion")
print("- Brightness represents magnitude (speed) of motion")

# Create a copy of frame1 in RGB for drawing arrows
frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB)

# Sampling step (draw arrows at every 'step' pixels)
step = 20

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Subplot 1: Arrow visualization
axes[0].imshow(frame1_rgb)
axes[0].set_title('Dense Optical Flow - Motion Vectors (Arrows)', fontsize=14)
axes[0].axis('off')

# Draw arrows for sampled points
for y in range(0, frame1.shape[0], step):
    for x in range(0, frame1.shape[1], step):
        # Get flow at this point
        fx, fy = flow[y, x]

        # Draw arrow from (x, y) to (x+fx, y+fy)
        # Color based on magnitude
        mag = magnitude[y, x]

        # Use color based on magnitude (red for high, blue for low)
        color = plt.cm.jet(mag / magnitude.max())

        axes[0].arrow(x, y, fx, fy,
                     color=color,
                     head_width=3,
                     head_length=3,
                     linewidth=1.5,
                     alpha=0.7)

# Subplot 2: Magnitude overlay for reference
axes[1].imshow(frame1, cmap='gray', alpha=0.5)
im = axes[1].imshow(magnitude, cmap='hot', alpha=0.6)
axes[1].set_title('Flow Magnitude Overlay', fontsize=14)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046, label='Magnitude (pixels)')

plt.tight_layout()
plt.savefig(output_folder / '03_motion_vectors_arrows.png', dpi=150, bbox_inches='tight')
print(f"Saved: 03_motion_vectors_arrows.png")
plt.show()

# Define thresholds for fast and slow motion
# Calculate percentiles to identify different motion regions
percentile_75 = np.percentile(magnitude, 75)
percentile_25 = np.percentile(magnitude, 25)

print(f"Magnitude Statistics:")
print(f"  Min: {magnitude.min():.2f} pixels")
print(f"  25th percentile: {percentile_25:.2f} pixels")
print(f"  Mean: {magnitude.mean():.2f} pixels")
print(f"  Median: {np.median(magnitude):.2f} pixels")
print(f"  75th percentile: {percentile_75:.2f} pixels")
print(f"  Max: {magnitude.max():.2f} pixels")

# Create masks for different motion regions
slow_motion_mask = magnitude < percentile_25
medium_motion_mask = (magnitude >= percentile_25) & (magnitude < percentile_75)
fast_motion_mask = magnitude >= percentile_75

# Calculate statistics for each region
slow_region_count = np.sum(slow_motion_mask)
medium_region_count = np.sum(medium_motion_mask)
fast_region_count = np.sum(fast_motion_mask)

print(f"\nRegion Distribution:")
print(f"  Slow-moving regions: {slow_region_count} pixels ({slow_region_count / magnitude.size * 100:.1f}%)")
print(f"  Medium-moving regions: {medium_region_count} pixels ({medium_region_count / magnitude.size * 100:.1f}%)")
print(f"  Fast-moving regions: {fast_region_count} pixels ({fast_region_count / magnitude.size * 100:.1f}%)")

# Visualize the different regions
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Original frame with all arrows
axes[0, 0].imshow(frame1, cmap='gray')
axes[0, 0].set_title('Original Frame', fontsize=14)
axes[0, 0].axis('off')

# Segmented regions by motion speed
segmented = np.zeros_like(frame1, dtype=np.uint8)
segmented[slow_motion_mask] = 85  # Dark gray for slow
segmented[medium_motion_mask] = 170  # Medium gray
segmented[fast_motion_mask] = 255  # White for fast

axes[0, 1].imshow(segmented, cmap='gray')
axes[0, 1].set_title('Motion Segmentation (Dark=Slow, Bright=Fast)', fontsize=14)
axes[0, 1].axis('off')

# Fast-moving regions with arrows
axes[1, 0].imshow(frame1_rgb)
axes[1, 0].set_title('Fast-Moving Regions (Motion Vectors)', fontsize=14)
axes[1, 0].axis('off')

for y in range(0, frame1.shape[0], step):
    for x in range(0, frame1.shape[1], step):
        if fast_motion_mask[y, x]:
            fx, fy = flow[y, x]
            axes[1, 0].arrow(x, y, fx, fy, color='red',
                           head_width=3, head_length=3, linewidth=2, alpha=0.8)

# Slow-moving regions with arrows
axes[1, 1].imshow(frame1_rgb)
axes[1, 1].set_title('Slow-Moving Regions (Motion Vectors)', fontsize=14)
axes[1, 1].axis('off')

for y in range(0, frame1.shape[0], step):
    for x in range(0, frame1.shape[1], step):
        if slow_motion_mask[y, x]:
            fx, fy = flow[y, x]
            axes[1, 1].arrow(x, y, fx, fy, color='blue',
                           head_width=3, head_length=3, linewidth=2, alpha=0.8)

plt.tight_layout()
plt.savefig(output_folder / '04_motion_segmentation.png', dpi=150, bbox_inches='tight')
print(f"Saved: 04_motion_segmentation.png")
plt.show()

print("\n" + "="*80)
print("All visualizations saved successfully!")
print(f"Output folder: {output_folder}")
print("="*80)
