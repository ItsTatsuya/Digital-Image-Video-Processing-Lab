import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

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
print("BLOCK-BASED STEREO MATCHING FOR OBSTACLE DETECTION")
print("="*80)

# Load images
left_img = cv2.imread(str(left_image_path), cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(str(right_image_path), cv2.IMREAD_GRAYSCALE)

print(f"\nLeft image: {left_image_path.name}")
print(f"Right image: {right_image_path.name}")
print(f"Image shape: {left_img.shape}")

# Parse calibration file
def parse_calibration(calib_file):
    """Parse KITTI calibration file to extract camera parameters."""
    with open(calib_file, 'r') as f:
        lines = f.readlines()

    P0 = np.array([float(x) for x in lines[0].strip().split()[1:]]).reshape(3, 4)
    P1 = np.array([float(x) for x in lines[1].strip().split()[1:]]).reshape(3, 4)

    focal_length = P0[0, 0]
    baseline = abs(P1[0, 3]) / focal_length

    return focal_length, baseline

focal_length, baseline = parse_calibration(calib_path)

print(f"\nCamera Parameters:")
print(f"  Focal length: {focal_length:.2f} pixels")
print(f"  Baseline: {baseline:.4f} meters ({baseline*100:.2f} cm)")

# Display stereo pair
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
axes[0].imshow(left_img, cmap='gray')
axes[0].set_title('Left Image (Reference)', fontsize=14)
axes[0].axis('off')

axes[1].imshow(right_img, cmap='gray')
axes[1].set_title('Right Image', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig(output_folder / '01_stereo_pair.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: 01_stereo_pair.png")
plt.show()

def compute_sad(block1, block2):
    """Compute Sum of Absolute Differences between two blocks."""
    return np.sum(np.abs(block1.astype(np.int16) - block2.astype(np.int16)))

def block_matching_stereo(left_img, right_img, block_size=16, max_disparity=128):
    """
    Perform block-based stereo matching using SAD along scanlines.

    Args:
        left_img: Left (reference) image
        right_img: Right image
        block_size: Size of blocks (default 16x16)
        max_disparity: Maximum disparity search range (default 128 pixels)

    Returns:
        disparity_map: Computed disparity map
    """
    h, w = left_img.shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    print(f"\nComputing disparity using block matching:")
    print(f"  Block size: {block_size}x{block_size}")
    print(f"  Max disparity: {max_disparity} pixels")

    start_time = time.time()
    block_count = 0

    # Process blocks
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            # Extract block from left image
            left_block = left_img[y:y+block_size, x:x+block_size]

            min_sad = float('inf')
            best_disparity = 0

            # Search along the same scanline (epipolar constraint)
            # In rectified stereo, corresponding points are on the same row
            # Search to the LEFT in the right image (negative x direction)
            search_start = max(0, x - max_disparity)
            search_end = x

            for search_x in range(search_start, search_end + 1):
                # Extract candidate block from right image
                right_block = right_img[y:y+block_size, search_x:search_x+block_size]

                # Compute SAD
                sad = compute_sad(left_block, right_block)

                if sad < min_sad:
                    min_sad = sad
                    best_disparity = x - search_x

            # Fill the disparity map for this block
            disparity_map[y:y+block_size, x:x+block_size] = best_disparity
            block_count += 1

        # Progress indicator
        if (y // block_size) % 5 == 0:
            progress = (y / h) * 100
            print(f"  Progress: {progress:.1f}%", end='\r')

    elapsed_time = time.time() - start_time
    print(f"\n  Processed {block_count} blocks in {elapsed_time:.2f} seconds")

    return disparity_map

# Compute disparity map
print("\n" + "="*80)
print("STEP 1: COMPUTING DISPARITY MAP")
print("="*80)

disparity_map = block_matching_stereo(left_img, right_img, block_size=16, max_disparity=128)

print(f"\nDisparity Map Statistics:")
print(f"  Shape: {disparity_map.shape}")
print(f"  Min disparity: {disparity_map.min():.2f} pixels")
print(f"  Max disparity: {disparity_map.max():.2f} pixels")
print(f"  Mean disparity: {disparity_map.mean():.2f} pixels")
print(f"  Median disparity: {np.median(disparity_map):.2f} pixels")

# Visualize disparity map
disparity_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
disparity_color_rgb = cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(24, 10))

axes[0].imshow(left_img, cmap='gray')
axes[0].set_title('Left Image (Reference)', fontsize=14)
axes[0].axis('off')

im = axes[1].imshow(disparity_color_rgb)
axes[1].set_title('Disparity Map (High=Close, Low=Far)', fontsize=14)
axes[1].axis('off')

cbar = plt.colorbar(axes[1].imshow(disparity_map, cmap='jet'), ax=axes[1], fraction=0.046)
cbar.set_label('Disparity (pixels)', fontsize=12)

plt.tight_layout()
plt.savefig(output_folder / '02_disparity_map.png', dpi=150, bbox_inches='tight')
print(f"Saved: 02_disparity_map.png")
plt.show()

# Compute depth map
print("\n" + "="*80)
print("STEP 2: CONVERTING DISPARITY TO DEPTH")
print("="*80)

# Avoid division by zero
disparity_safe = disparity_map.copy()
disparity_safe[disparity_safe == 0] = 0.1

# Depth = (focal_length * baseline) / disparity
depth_map = (focal_length * baseline) / disparity_safe

# Clip to reasonable range
max_depth = 100.0  # meters
depth_map = np.clip(depth_map, 0, max_depth)

print(f"\nDepth Map Statistics:")
print(f"  Min depth: {depth_map.min():.2f} meters")
print(f"  Max depth: {depth_map.max():.2f} meters")
print(f"  Mean depth: {depth_map.mean():.2f} meters")
print(f"  Median depth: {np.median(depth_map):.2f} meters")

# Visualize depth map
depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_color = cv2.applyColorMap(255 - depth_normalized, cv2.COLORMAP_JET)
depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(24, 10))

axes[0].imshow(disparity_color_rgb)
axes[0].set_title('Disparity Map', fontsize=14)
axes[0].axis('off')

im = axes[1].imshow(depth_color_rgb)
axes[1].set_title('Depth Map (Red=Close, Blue=Far)', fontsize=14)
axes[1].axis('off')

cbar = plt.colorbar(axes[1].imshow(depth_map, cmap='jet_r'), ax=axes[1], fraction=0.046)
cbar.set_label('Depth (meters)', fontsize=12)

plt.tight_layout()
plt.savefig(output_folder / '03_depth_map.png', dpi=150, bbox_inches='tight')
print(f"Saved: 03_depth_map.png")
plt.show()

# Obstacle detection with different thresholds
print("\n" + "="*80)
print("STEP 3: OBSTACLE DETECTION WITH VARYING DEPTH THRESHOLDS")
print("="*80)

# Test multiple depth thresholds
depth_thresholds = [5.0, 10.0, 15.0, 20.0]

fig, axes = plt.subplots(2, 2, figsize=(24, 16))
axes = axes.flatten()

obstacle_results = {}

for idx, threshold in enumerate(depth_thresholds):
    # Detect obstacles (objects closer than threshold)
    obstacle_mask = depth_map < threshold

    # Calculate statistics
    obstacle_count = np.sum(obstacle_mask)
    obstacle_percentage = (obstacle_count / depth_map.size) * 100

    obstacle_results[threshold] = {
        'count': obstacle_count,
        'percentage': obstacle_percentage,
        'mask': obstacle_mask
    }

    print(f"\nDepth Threshold: {threshold:.1f}m")
    print(f"  Obstacles detected: {obstacle_count} pixels ({obstacle_percentage:.2f}%)")

    # Visualize
    axes[idx].imshow(left_img, cmap='gray')

    # Overlay obstacles in red
    obstacle_overlay = np.zeros((*left_img.shape, 3), dtype=np.uint8)
    obstacle_overlay[obstacle_mask] = [255, 0, 0]
    axes[idx].imshow(obstacle_overlay, alpha=0.5)

    axes[idx].set_title(f'Obstacles < {threshold:.1f}m ({obstacle_percentage:.1f}% of image)', fontsize=14)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(output_folder / '04_obstacle_detection_thresholds.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: 04_obstacle_detection_thresholds.png")
plt.show()

# Detailed analysis of obstacle detection
print("\n" + "="*80)
print("DETAILED OBSTACLE ANALYSIS")
print("="*80)

# Use a standard threshold for detailed analysis
standard_threshold = 10.0  # meters
obstacle_mask = depth_map < standard_threshold

# Classify regions by depth
near_mask = depth_map < 5.0
medium_mask = (depth_map >= 5.0) & (depth_map < 15.0)
far_mask = depth_map >= 15.0

near_count = np.sum(near_mask)
medium_count = np.sum(medium_mask)
far_count = np.sum(far_mask)

print(f"\nDepth Classification (standard threshold = {standard_threshold}m):")
print(f"  CRITICAL (< 5m): {near_count} pixels ({near_count / depth_map.size * 100:.1f}%)")
print(f"  WARNING (5-15m): {medium_count} pixels ({medium_count / depth_map.size * 100:.1f}%)")
print(f"  SAFE (> 15m): {far_count} pixels ({far_count / depth_map.size * 100:.1f}%)")

# Visualize detailed classification
fig, axes = plt.subplots(2, 3, figsize=(24, 16))

# Row 1
axes[0, 0].imshow(left_img, cmap='gray')
axes[0, 0].set_title('Original Left Image', fontsize=14)
axes[0, 0].axis('off')

axes[0, 1].imshow(depth_color_rgb)
axes[0, 1].set_title('Depth Map', fontsize=14)
axes[0, 1].axis('off')

# Depth segmentation
depth_segmented = np.zeros_like(depth_map)
depth_segmented[near_mask] = 1
depth_segmented[medium_mask] = 2
depth_segmented[far_mask] = 3

axes[0, 2].imshow(depth_segmented, cmap='RdYlGn_r')
axes[0, 2].set_title('Depth Zones (Red=Critical, Yellow=Warning, Green=Safe)', fontsize=14)
axes[0, 2].axis('off')

# Row 2: Individual zone highlights
axes[1, 0].imshow(left_img, cmap='gray')
overlay = np.zeros((*left_img.shape, 3), dtype=np.uint8)
overlay[near_mask] = [255, 0, 0]
axes[1, 0].imshow(overlay, alpha=0.6)
axes[1, 0].set_title(f'CRITICAL Zone (< 5m) - {near_count / depth_map.size * 100:.1f}%', fontsize=14)
axes[1, 0].axis('off')

axes[1, 1].imshow(left_img, cmap='gray')
overlay = np.zeros((*left_img.shape, 3), dtype=np.uint8)
overlay[medium_mask] = [255, 255, 0]
axes[1, 1].imshow(overlay, alpha=0.6)
axes[1, 1].set_title(f'WARNING Zone (5-15m) - {medium_count / depth_map.size * 100:.1f}%', fontsize=14)
axes[1, 1].axis('off')

axes[1, 2].imshow(left_img, cmap='gray')
overlay = np.zeros((*left_img.shape, 3), dtype=np.uint8)
overlay[far_mask] = [0, 255, 0]
axes[1, 2].imshow(overlay, alpha=0.6)
axes[1, 2].set_title(f'SAFE Zone (> 15m) - {far_count / depth_map.size * 100:.1f}%', fontsize=14)
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(output_folder / '05_detailed_obstacle_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: 05_detailed_obstacle_analysis.png")
plt.show()

# Comparison: Our implementation vs OpenCV StereoSGBM
print("\n" + "="*80)
print("COMPARISON: BLOCK MATCHING vs STEREOSGBM")
print("="*80)

print("\nComputing StereoSGBM for comparison...")
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=11,
    P1=8 * 3 * 11 ** 2,
    P2=32 * 3 * 11 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity_sgbm = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
disparity_sgbm[disparity_sgbm <= 0] = 0.1

depth_sgbm = (focal_length * baseline) / disparity_sgbm
depth_sgbm = np.clip(depth_sgbm, 0, max_depth)

print(f"\nStereoSGBM Depth Statistics:")
print(f"  Min: {depth_sgbm.min():.2f}m, Max: {depth_sgbm.max():.2f}m")
print(f"  Mean: {depth_sgbm.mean():.2f}m, Median: {np.median(depth_sgbm):.2f}m")

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(24, 16))

# Block matching results
disparity_bm_vis = cv2.applyColorMap(
    cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    cv2.COLORMAP_JET
)
axes[0, 0].imshow(cv2.cvtColor(disparity_bm_vis, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Disparity Map - Block Matching (Our Implementation)', fontsize=14)
axes[0, 0].axis('off')

depth_bm_vis = cv2.applyColorMap(
    cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    cv2.COLORMAP_JET
)
axes[0, 1].imshow(cv2.cvtColor(depth_bm_vis, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Depth Map - Block Matching', fontsize=14)
axes[0, 1].axis('off')

# StereoSGBM results
disparity_sgbm_vis = cv2.applyColorMap(
    cv2.normalize(disparity_sgbm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    cv2.COLORMAP_JET
)
axes[1, 0].imshow(cv2.cvtColor(disparity_sgbm_vis, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Disparity Map - StereoSGBM (OpenCV)', fontsize=14)
axes[1, 0].axis('off')

depth_sgbm_vis = cv2.applyColorMap(
    cv2.normalize(depth_sgbm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    cv2.COLORMAP_JET
)
axes[1, 1].imshow(cv2.cvtColor(depth_sgbm_vis, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Depth Map - StereoSGBM', fontsize=14)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(output_folder / '06_comparison_sgbm.png', dpi=150, bbox_inches='tight')
print(f"Saved: 06_comparison_sgbm.png")
plt.show()

# Print comprehensive commentary
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS AND COMMENTARY")
print("="*80)

print("\n(a) WHICH OBJECTS ARE DETECTED AS OBSTACLES AND WHY:")
print("-" * 80)
print(f"""
OBSTACLE DETECTION USING DEPTH THRESHOLD ({standard_threshold}m):

DETECTED OBSTACLES ({near_count / depth_map.size * 100:.1f}% of image):

1. VEHICLES DIRECTLY AHEAD:
   WHY DETECTED AS OBSTACLES:
   - High disparity values (large pixel shift in stereo pair)
   - Objects are 2-10 meters from camera
   - Clear depth discontinuity from background
   - High texture (easy to match blocks)
   - Represent IMMEDIATE COLLISION HAZARD

   CHARACTERISTICS:
   ✓ Strong stereo correspondence
   ✓ Well-defined edges and boundaries
   ✓ Consistent disparity within object
   ✓ Depth significantly less than threshold

2. ROAD SURFACE (NEAR FIELD):
   WHY DETECTED:
   - Ground plane in front of vehicle
   - Distance decreases with proximity to camera
   - Visible at bottom of image (0-5 meters)
   - Part of drivable surface, not truly an obstacle

   NOTE: Road surface detection is GOOD for:
   - Free space estimation
   - Ground plane modeling
   - But should be filtered for obstacle detection

3. ROADSIDE OBJECTS:
   WHY DETECTED:
   - Curbs, barriers, poles within {standard_threshold}m
   - High disparity due to proximity
   - Potential obstacles if vehicle deviates
   - Important for lane keeping and path planning

4. NEARBY PEDESTRIANS/CYCLISTS (if present):
   WHY DETECTED:
   - Critical safety concern
   - Small objects with high disparity
   - Dynamic obstacles requiring immediate response
   - High priority for autonomous driving

NOT DETECTED AS OBSTACLES:

1. DISTANT VEHICLES (> {standard_threshold}m):
   - Low disparity values
   - Beyond immediate collision risk
   - Useful for long-term planning, not immediate avoidance

2. BACKGROUND BUILDINGS/SCENERY:
   - Very low disparity (far away)
   - Not in vehicle's path
   - Depth >> threshold

3. SKY:
   - Effectively infinite distance
   - Zero or very low disparity
   - No physical obstacle
   - Often has poor stereo matching

DETECTION RELIABILITY FACTORS:

HIGH CONFIDENCE OBSTACLES:
  ✓ High disparity (> {disparity_map.mean():.0f} pixels)
  ✓ Textured surfaces (good block matching)
  ✓ Well-lit regions
  ✓ Clear depth boundaries
  ✓ Consistent within object

LOW CONFIDENCE OBSTACLES:
  ✗ Low texture (uniform surfaces)
  ✗ Shadows and dark regions
  ✗ Reflective surfaces (windows, wet road)
  ✗ Thin objects (poles, wires)
  ✗ Occluded regions (visible in only one camera)

BLOCK MATCHING PERFORMANCE:

ADVANTAGES:
  + Simple and efficient algorithm
  + Works well for textured surfaces
  + Natural scanline constraint (epipolar geometry)
  + Real-time capable
  + Good for dominant obstacles

LIMITATIONS:
  - Block size (16x16) may miss small objects
  - Uniform blocks lead to ambiguous matches
  - No sub-pixel accuracy
  - Blocky artifacts at boundaries
  - Assumes Lambertian surfaces (constant brightness)
""")

print("\n(b) HOW CHANGING DEPTH THRESHOLD AFFECTS DETECTION RESULTS:")
print("-" * 80)
print(f"""
THRESHOLD ANALYSIS:

THRESHOLD = 5.0m (VERY CONSERVATIVE):
  Detected: {obstacle_results[5.0]['percentage']:.2f}% of image

  EFFECT:
  ✓ Only IMMEDIATE obstacles detected
  ✓ Very high confidence detections
  ✓ Minimal false positives
  ✓ Critical collision zone only

  USE CASES:
  - Emergency braking systems
  - Last-resort collision avoidance
  - Low-speed urban environments
  - Parking assistance

  LIMITATIONS:
  - Very short reaction time
  - No lookahead for planning
  - May miss approaching obstacles too late

THRESHOLD = 10.0m (MODERATE):
  Detected: {obstacle_results[10.0]['percentage']:.2f}% of image

  EFFECT:
  ✓ Immediate + near-term obstacles
  ✓ Reasonable reaction time at urban speeds
  ✓ Good balance of coverage and false positives
  ✓ Standard autonomous driving threshold

  USE CASES:
  - City driving (30-50 km/h)
  - Adaptive cruise control
  - Lane keeping assistance
  - Standard obstacle avoidance

  OPTIMAL FOR:
  - Most autonomous driving scenarios
  - Balances safety and efficiency

THRESHOLD = 15.0m (MODERATE-AGGRESSIVE):
  Detected: {obstacle_results[15.0]['percentage']:.2f}% of image

  EFFECT:
  ✓ Extended detection range
  ✓ More planning time
  ✓ Captures medium-distance objects
  ⚠ Increased false positive risk

  USE CASES:
  - Highway driving (moderate speed)
  - Path planning with lookahead
  - Comfortable braking distances

  TRADE-OFF:
  - More detections = more processing
  - May include irrelevant objects
  - Better for proactive systems

THRESHOLD = 20.0m (AGGRESSIVE):
  Detected: {obstacle_results[20.0]['percentage']:.2f}% of image

  EFFECT:
  ✓ Very long detection range
  ✓ Maximum planning horizon
  ⚠ Many non-critical detections
  ⚠ Higher computational load

  USE CASES:
  - High-speed highway driving
  - Advanced path planning
  - Multi-lane scenarios
  - Long-term trajectory optimization

  CHALLENGES:
  - Lower depth accuracy at far distances
  - Many objects may not be in path
  - Increased false positive rate
  - Need sophisticated filtering

RELATIONSHIP: THRESHOLD vs DETECTION PERCENTAGE

As threshold increases:
  ↑ More pixels classified as obstacles (exponentially)
  ↑ Detection coverage increases
  ↑ False positive rate increases
  ↑ Computational cost increases
  ↓ Detection confidence may decrease
  ↓ Precision decreases (more noise)

INVERSE RELATIONSHIP WITH DISPARITY:
  - Small threshold → High disparity required → Few detections
  - Large threshold → Low disparity accepted → Many detections
  - Due to: Depth = (f × B) / disparity

PRACTICAL RECOMMENDATIONS:

URBAN DRIVING (< 50 km/h):
  Recommended threshold: 8-12 meters
  Reasoning: Short braking distances, close obstacles

HIGHWAY DRIVING (> 80 km/h):
  Recommended threshold: 20-30 meters
  Reasoning: Long braking distances, need lookahead

PARKING:
  Recommended threshold: 2-5 meters
  Reasoning: Very low speed, immediate surroundings only

ADAPTIVE THRESHOLDING:
  BEST APPROACH: Dynamic threshold based on:
  - Current vehicle speed
  - Road type (urban vs highway)
  - Weather conditions
  - System response time

  Formula: threshold = (speed² / 2a) + safety_margin
  Where: a = maximum deceleration
         safety_margin = 2-5 meters

MULTI-ZONE APPROACH:
  Instead of single threshold, use zones:

  ZONE 1 (< 5m): CRITICAL
    → Emergency braking
    → Maximum priority
    → Immediate action required

  ZONE 2 (5-15m): WARNING
    → Prepare to brake
    → Path planning
    → High priority

  ZONE 3 (15-30m): MONITORING
    → Long-term planning
    → Lane changes
    → Medium priority

  ZONE 4 (> 30m): AWARENESS
    → Context awareness
    → Behavior prediction
    → Low priority

FILTERING STRATEGIES:

To improve detection quality:
  1. Ground plane removal (filter out road surface)
  2. Height filtering (objects must be above ground)
  3. Temporal filtering (track objects across frames)
  4. Size filtering (minimum object size threshold)
  5. Confidence filtering (SAD value threshold)

SUMMARY:
  - Threshold choice is CRITICAL for safety vs efficiency
  - No single "best" threshold - depends on context
  - Adaptive/multi-zone approaches are superior
  - Must balance reaction time vs false positives
  - Higher speed → larger threshold required
""")

print("\n" + "="*80)
print("All visualizations saved successfully!")
print(f"Output folder: {output_folder}")
print("="*80)
