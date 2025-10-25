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
image_0_folder = dataset_path / 'training' / 'image_0'  # Left camera
image_1_folder = dataset_path / 'training' / 'image_1'  # Right camera
calib_folder = dataset_path / 'training' / 'calib'
output_folder = script_dir / 'output'
output_folder.mkdir(exist_ok=True)

# Select a sequence
sequence_idx = 10

# Load stereo pair
left_image_path = image_0_folder / f'{sequence_idx:06d}_10.png'
right_image_path = image_1_folder / f'{sequence_idx:06d}_10.png'
calib_path = calib_folder / f'{sequence_idx:06d}.txt'

print("="*80)
print("STEREO VISION - DISPARITY AND DEPTH ESTIMATION")
print("="*80)

# Load images
left_img = cv2.imread(str(left_image_path), cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(str(right_image_path), cv2.IMREAD_GRAYSCALE)

print(f"\nLeft image: {left_image_path.name}")
print(f"Right image: {right_image_path.name}")
print(f"Image shape: {left_img.shape}")

# Parse calibration file to extract camera parameters
def parse_calibration(calib_file):
    """Parse KITTI calibration file to extract camera parameters."""
    with open(calib_file, 'r') as f:
        lines = f.readlines()

    # Parse P0 (left camera projection matrix)
    P0 = np.array([float(x) for x in lines[0].strip().split()[1:]]).reshape(3, 4)
    # Parse P1 (right camera projection matrix)
    P1 = np.array([float(x) for x in lines[1].strip().split()[1:]]).reshape(3, 4)

    # Extract focal length (assuming fx = fy)
    focal_length = P0[0, 0]

    # Extract baseline from P1
    # baseline = -P1[0, 3] / P1[0, 0]
    # For KITTI, the baseline is approximately 0.54 meters
    baseline = abs(P1[0, 3]) / focal_length

    # Principal point
    cx = P0[0, 2]
    cy = P0[1, 2]

    return focal_length, baseline, cx, cy, P0, P1

focal_length, baseline, cx, cy, P0, P1 = parse_calibration(calib_path)

print(f"\nCamera Parameters:")
print(f"  Focal length: {focal_length:.2f} pixels")
print(f"  Baseline: {baseline:.4f} meters ({baseline*100:.2f} cm)")
print(f"  Principal point: ({cx:.2f}, {cy:.2f})")
print(f"  P0 (Left camera):")
print(f"    {P0}")
print(f"  P1 (Right camera):")
print(f"    {P1}")

# Display stereo pair
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
axes[0].imshow(left_img, cmap='gray')
axes[0].set_title('Left Image (Camera 0)', fontsize=14)
axes[0].axis('off')

axes[1].imshow(right_img, cmap='gray')
axes[1].set_title('Right Image (Camera 1)', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig(output_folder / '01_stereo_pair.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: 01_stereo_pair.png")
plt.show()

# Compute disparity map using StereoSGBM
print("\n" + "="*80)
print("COMPUTING DISPARITY MAP USING STEREOSGBM")
print("="*80)

# StereoSGBM parameters
min_disparity = 0
num_disparities = 128  # Must be divisible by 16
block_size = 11  # Odd number, typically 3-11

# Create StereoSGBM object
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,  # Smoothness parameter 1
    P2=32 * 3 * block_size ** 2,  # Smoothness parameter 2
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

print(f"\nStereoSGBM Parameters:")
print(f"  Min disparity: {min_disparity}")
print(f"  Num disparities: {num_disparities}")
print(f"  Block size: {block_size}")
print(f"  P1: {8 * 3 * block_size ** 2}")
print(f"  P2: {32 * 3 * block_size ** 2}")

# Compute disparity
print("\nComputing disparity map...")
disparity = stereo.compute(left_img, right_img)

# Convert to float and scale (StereoSGBM returns values scaled by 16)
disparity_map = disparity.astype(np.float32) / 16.0

# Filter out invalid disparities
disparity_map[disparity_map <= 0] = 0.1  # Avoid division by zero

print(f"\nDisparity Map Statistics:")
print(f"  Shape: {disparity_map.shape}")
print(f"  Data type: {disparity_map.dtype}")
print(f"  Min disparity: {disparity_map.min():.2f} pixels")
print(f"  Max disparity: {disparity_map.max():.2f} pixels")
print(f"  Mean disparity: {disparity_map.mean():.2f} pixels")
print(f"  Median disparity: {np.median(disparity_map):.2f} pixels")

# Normalize disparity for visualization
disparity_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
disparity_color_rgb = cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB)

# Visualize disparity map
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

axes[0].imshow(left_img, cmap='gray')
axes[0].set_title('Left Image (Reference)', fontsize=14)
axes[0].axis('off')

im = axes[1].imshow(disparity_color_rgb)
axes[1].set_title('Disparity Map (Closer=Red/Yellow, Farther=Blue)', fontsize=14)
axes[1].axis('off')

# Add colorbar
cbar = plt.colorbar(axes[1].imshow(disparity_map, cmap='jet'), ax=axes[1], fraction=0.046)
cbar.set_label('Disparity (pixels)', fontsize=12)

plt.tight_layout()
plt.savefig(output_folder / '02_disparity_map.png', dpi=150, bbox_inches='tight')
print(f"Saved: 02_disparity_map.png")
plt.show()

# Compute depth map
print("\n" + "="*80)
print("COMPUTING DEPTH MAP FROM DISPARITY")
print("="*80)

# Depth = (focal_length * baseline) / disparity
# baseline is in meters, so depth will be in meters
depth_map = (focal_length * baseline) / (disparity_map + 1e-6)  # Add small value to avoid division by zero

# Clip depth values to reasonable range (e.g., 0 to 100 meters)
max_depth = 100.0  # meters
depth_map_clipped = np.clip(depth_map, 0, max_depth)

print(f"\nDepth Map Statistics:")
print(f"  Shape: {depth_map_clipped.shape}")
print(f"  Min depth: {depth_map_clipped.min():.2f} meters")
print(f"  Max depth: {depth_map_clipped.max():.2f} meters (clipped at {max_depth}m)")
print(f"  Mean depth: {depth_map_clipped.mean():.2f} meters")
print(f"  Median depth: {np.median(depth_map_clipped):.2f} meters")

# Normalize depth for visualization (inverse for better visualization - closer is brighter)
depth_normalized = cv2.normalize(depth_map_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_color = cv2.applyColorMap(255 - depth_normalized, cv2.COLORMAP_JET)  # Invert for closer=warm colors
depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

# Visualize depth map
fig, axes = plt.subplots(2, 2, figsize=(24, 16))

# Row 1: Disparity and Depth maps
axes[0, 0].imshow(disparity_color_rgb)
axes[0, 0].set_title('Disparity Map (pixels)', fontsize=14)
axes[0, 0].axis('off')

axes[0, 1].imshow(depth_color_rgb)
axes[0, 1].set_title('Depth Map (meters) - Red=Close, Blue=Far', fontsize=14)
axes[0, 1].axis('off')

# Row 2: Depth map with different visualizations
im1 = axes[1, 0].imshow(depth_map_clipped, cmap='jet_r')  # _r for reverse (close=warm)
axes[1, 0].set_title('Depth Map (Continuous)', fontsize=14)
axes[1, 0].axis('off')
cbar1 = plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
cbar1.set_label('Depth (meters)', fontsize=12)

# Overlay on left image
axes[1, 1].imshow(left_img, cmap='gray', alpha=0.5)
im2 = axes[1, 1].imshow(depth_map_clipped, cmap='jet_r', alpha=0.6)
axes[1, 1].set_title('Depth Overlay on Left Image', fontsize=14)
axes[1, 1].axis('off')
cbar2 = plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
cbar2.set_label('Depth (meters)', fontsize=12)

plt.tight_layout()
plt.savefig(output_folder / '03_depth_map.png', dpi=150, bbox_inches='tight')
print(f"Saved: 03_depth_map.png")
plt.show()

# Analyze depth distribution
print("\n" + "="*80)
print("DEPTH DISTRIBUTION ANALYSIS")
print("="*80)

# Define depth zones
near_threshold = 10.0  # meters
medium_threshold = 30.0  # meters

near_mask = depth_map_clipped < near_threshold
medium_mask = (depth_map_clipped >= near_threshold) & (depth_map_clipped < medium_threshold)
far_mask = depth_map_clipped >= medium_threshold

near_count = np.sum(near_mask)
medium_count = np.sum(medium_mask)
far_count = np.sum(far_mask)

print(f"\nDepth Zones:")
print(f"  Near (< {near_threshold}m): {near_count} pixels ({near_count / depth_map_clipped.size * 100:.1f}%)")
print(f"  Medium ({near_threshold}-{medium_threshold}m): {medium_count} pixels ({medium_count / depth_map_clipped.size * 100:.1f}%)")
print(f"  Far (> {medium_threshold}m): {far_count} pixels ({far_count / depth_map_clipped.size * 100:.1f}%)")

# Visualize depth zones
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Depth segmentation
depth_segmented = np.zeros_like(depth_map_clipped)
depth_segmented[near_mask] = 1
depth_segmented[medium_mask] = 2
depth_segmented[far_mask] = 3

axes[0, 0].imshow(left_img, cmap='gray')
axes[0, 0].set_title('Original Left Image', fontsize=14)
axes[0, 0].axis('off')

axes[0, 1].imshow(depth_segmented, cmap='viridis')
axes[0, 1].set_title(f'Depth Zones (Near<{near_threshold}m, Medium<{medium_threshold}m, Far>{medium_threshold}m)', fontsize=14)
axes[0, 1].axis('off')

# Highlight near objects
axes[1, 0].imshow(left_img, cmap='gray')
near_overlay = np.zeros((*left_img.shape, 3), dtype=np.uint8)
near_overlay[near_mask] = [255, 0, 0]  # Red for near objects
axes[1, 0].imshow(near_overlay, alpha=0.5)
axes[1, 0].set_title(f'Near Objects (< {near_threshold}m) - Red Overlay', fontsize=14)
axes[1, 0].axis('off')

# Histogram of depth values
axes[1, 1].hist(depth_map_clipped.flatten(), bins=100, color='blue', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(near_threshold, color='red', linestyle='--', linewidth=2, label=f'Near threshold ({near_threshold}m)')
axes[1, 1].axvline(medium_threshold, color='orange', linestyle='--', linewidth=2, label=f'Medium threshold ({medium_threshold}m)')
axes[1, 1].set_xlabel('Depth (meters)', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Depth Distribution', fontsize=14)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_folder / '04_depth_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: 04_depth_analysis.png")
plt.show()

# Relationship between disparity and depth
print("\n" + "="*80)
print("DISPARITY-DEPTH RELATIONSHIP ANALYSIS")
print("="*80)

# Sample points for scatter plot
sample_size = min(10000, disparity_map.size)
sample_indices = np.random.choice(disparity_map.size, size=sample_size, replace=False)
disparity_samples = disparity_map.flatten()[sample_indices]
depth_samples = depth_map_clipped.flatten()[sample_indices]

# Filter out invalid values
valid_mask = (disparity_samples > 0) & (depth_samples < max_depth)
disparity_samples = disparity_samples[valid_mask]
depth_samples = depth_samples[valid_mask]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Scatter plot: Disparity vs Depth
axes[0].scatter(disparity_samples, depth_samples, alpha=0.3, s=1, c='blue')
axes[0].set_xlabel('Disparity (pixels)', fontsize=12)
axes[0].set_ylabel('Depth (meters)', fontsize=12)
axes[0].set_title('Disparity vs Depth Relationship (Inverse)', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Theoretical curve
disp_range = np.linspace(disparity_samples.min(), disparity_samples.max(), 100)
depth_theoretical = (focal_length * baseline) / disp_range
axes[0].plot(disp_range, depth_theoretical, 'r-', linewidth=2, label='Theoretical: Depth = (f×B)/d')
axes[0].legend()

# Inverse relationship
axes[1].scatter(depth_samples, disparity_samples, alpha=0.3, s=1, c='green')
axes[1].set_xlabel('Depth (meters)', fontsize=12)
axes[1].set_ylabel('Disparity (pixels)', fontsize=12)
axes[1].set_title('Depth vs Disparity Relationship', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Theoretical curve
depth_range = np.linspace(depth_samples.min(), depth_samples.max(), 100)
disp_theoretical = (focal_length * baseline) / depth_range
axes[1].plot(depth_range, disp_theoretical, 'r-', linewidth=2, label='Theoretical: d = (f×B)/Depth')
axes[1].legend()

plt.tight_layout()
plt.savefig(output_folder / '05_disparity_depth_relationship.png', dpi=150, bbox_inches='tight')
print(f"Saved: 05_disparity_depth_relationship.png")
plt.show()

# Print comprehensive commentary
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS AND COMMENTARY")
print("="*80)

print("\n(a) HOW DISPARITY VALUES RELATE TO OBJECT DISTANCE:")
print("-" * 80)
print(f"""
MATHEMATICAL RELATIONSHIP:
  Depth = (Focal_Length × Baseline) / Disparity

  Where:
    - Focal length (f) = {focal_length:.2f} pixels
    - Baseline (B) = {baseline:.4f} meters ({baseline*100:.2f} cm)
    - Disparity (d) = horizontal pixel shift between left and right images
    - Depth (Z) = distance from camera in meters

INVERSE RELATIONSHIP:
  ✓ High disparity → Close objects → Small depth
  ✓ Low disparity → Far objects → Large depth

  This is an INVERSE (hyperbolic) relationship:
  - Disparity decreases rapidly as distance increases
  - Small changes in disparity at far distances = large depth changes
  - Large changes in disparity at near distances = small depth changes

OBSERVED DATA:
  Disparity range: {disparity_map.min():.2f} to {disparity_map.max():.2f} pixels
  Corresponding depth: {depth_map_clipped.min():.2f}m to {depth_map_clipped.max():.2f}m

  Example calculations:
    - Disparity = {disparity_map.max():.2f} pixels → Depth = {(focal_length * baseline) / disparity_map.max():.2f}m (CLOSE)
    - Disparity = {np.median(disparity_map[disparity_map > 0]):.2f} pixels → Depth = {(focal_length * baseline) / np.median(disparity_map[disparity_map > 0]):.2f}m (MEDIUM)
    - Disparity = {disparity_map[disparity_map > 0].min():.2f} pixels → Depth = {(focal_length * baseline) / disparity_map[disparity_map > 0].min():.2f}m (FAR)

PHYSICAL INTERPRETATION:
  1. CLOSE OBJECTS (high disparity):
     - Large pixel shift between left and right images
     - Easy to detect and match
     - High depth accuracy
     - Objects appear at different positions in stereo pair

  2. FAR OBJECTS (low disparity):
     - Small pixel shift between left and right images
     - Harder to detect reliably
     - Lower depth accuracy (small errors → large depth changes)
     - Objects appear at nearly same position in stereo pair

  3. INFINITE DISTANCE (zero disparity):
     - No pixel shift
     - Both cameras see object at same position
     - Mathematically: Depth → ∞ as Disparity → 0

STEREO GEOMETRY:
  - Baseline ({baseline*100:.2f} cm) determines depth range and accuracy
  - Larger baseline → better accuracy for far objects
  - Smaller baseline → better for close objects, less occlusion
  - KITTI uses moderate baseline for automotive applications
""")

print("\n(b) WHICH REGIONS APPEAR CLOSER OR FARTHER IN THE DEPTH MAP:")
print("-" * 80)
print(f"""
NEAR REGIONS (< {near_threshold}m) - {near_count / depth_map_clipped.size * 100:.1f}% of image:

  CHARACTERISTICS:
    ✓ Appear in RED/YELLOW/WARM colors in visualization
    ✓ High disparity values
    ✓ Typically in foreground/bottom of image

  OBJECTS DETECTED AS CLOSE:
    1. VEHICLES IN FRONT:
       - Cars, trucks directly ahead
       - Typically 2-10 meters away
       - High confidence detection
       - Important for collision avoidance

    2. ROAD SURFACE:
       - Immediate roadway in front of camera
       - Ground plane at bottom of image
       - Known geometry aids estimation

    3. NEARBY OBSTACLES:
       - Pedestrians, cyclists
       - Roadside objects (poles, signs)
       - Curbs and barriers

    4. CAMERA MOUNTING VEHICLE:
       - Hood or front of ego vehicle may be visible
       - Very close (< 2m)

MEDIUM DISTANCE REGIONS ({near_threshold}-{medium_threshold}m) - {medium_count / depth_map_clipped.size * 100:.1f}% of image:

  CHARACTERISTICS:
    ✓ Appear in GREEN/CYAN colors
    ✓ Moderate disparity values
    ✓ Middle ground of scene

  OBJECTS DETECTED:
    1. TRAFFIC PARTICIPANTS:
       - Vehicles in adjacent lanes
       - Objects 10-30 meters ahead
       - Still trackable for planning

    2. ROAD INFRASTRUCTURE:
       - Traffic lights, signs at medium distance
       - Buildings along roadside
       - Trees and vegetation

    3. EXTENDED ROAD SURFACE:
       - Road extending forward
       - Visible lane markings

FAR REGIONS (> {medium_threshold}m) - {far_count / depth_map_clipped.size * 100:.1f}% of image:

  CHARACTERISTICS:
    ✓ Appear in BLUE/PURPLE/COOL colors
    ✓ Low disparity values (approaching zero)
    ✓ Background and upper portion of image
    ✓ Lower depth accuracy

  OBJECTS DETECTED AS FAR:
    1. SKY:
       - Effectively infinite distance
       - Very low/zero disparity
       - Often poorly defined

    2. DISTANT BACKGROUND:
       - Buildings far in distance
       - Horizon line
       - Mountains, clouds

    3. FAR ROAD SURFACE:
       - Road extending to vanishing point
       - 30+ meters ahead
       - Useful for long-term planning

    4. DISTANT VEHICLES:
       - Cars far ahead on highway
       - Too far for immediate concerns
       - May be tracked for behavior prediction

SPATIAL DISTRIBUTION:

  VERTICAL GRADIENT:
    - Bottom of image (road) → NEAR (warm colors)
    - Middle of image (scene) → MEDIUM (green)
    - Top of image (sky) → FAR (cool colors)
    - This follows natural perspective geometry

  OBSTACLES VS BACKGROUND:
    - Foreground objects → High disparity → Warm colors
    - Background → Low disparity → Cool colors
    - Clear segmentation between obstacle and free space

DEPTH MAP QUALITY INDICATORS:

  ✓ GOOD REGIONS:
    - Textured surfaces (easy to match)
    - Well-lit areas
    - Non-reflective surfaces
    - Areas within disparity range

  ✗ POOR REGIONS:
    - Sky (no texture, infinite depth)
    - Reflective surfaces (windows, wet road)
    - Shadows and dark areas
    - Occluded regions (visible in only one camera)
    - Repetitive patterns (matching ambiguity)
    - Very far objects (disparity → 0)

AUTOMOTIVE APPLICATION INSIGHTS:

  OBSTACLE DETECTION:
    - Objects < {near_threshold}m → CRITICAL (immediate hazard)
    - Objects {near_threshold}-{medium_threshold}m → WARNING (upcoming obstacle)
    - Objects > {medium_threshold}m → MONITORING (long-term planning)

  FREE SPACE DETECTION:
    - Road surface depth is continuous and smooth
    - Sudden depth discontinuities → obstacles
    - Horizontal surfaces at expected depth → drivable area

  LIMITATIONS:
    - Cannot detect objects beyond ~{max_depth}m reliably
    - Baseline limits maximum detectable distance
    - Requires texture for matching (fails on uniform surfaces)
    - Occlusions create depth estimation errors

COMPARISON WITH GROUND TRUTH:
  - KITTI dataset includes ground truth disparity/depth
  - This can be used to validate StereoSGBM results
  - Typical accuracy: 1-5% error for near objects
  - Accuracy degrades with distance (inverse relationship)
""")

print("\n" + "="*80)
print("All visualizations saved successfully!")
print(f"Output folder: {output_folder}")
print("="*80)
