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

def compute_sad(block1, block2):
    """
    Compute Sum of Absolute Differences (SAD) between two blocks.

    Args:
        block1: First block (reference)
        block2: Second block (candidate)

    Returns:
        SAD value (lower is better match)
    """
    return np.sum(np.abs(block1.astype(np.int16) - block2.astype(np.int16)))

def block_matching_sad(frame1, frame2, block_size=16, search_range=32):
    """
    Perform block matching using SAD (Sum of Absolute Differences).

    Args:
        frame1: Reference frame (previous frame)
        frame2: Current frame
        block_size: Size of blocks (default 16x16)
        search_range: Maximum search distance in pixels (default 32)

    Returns:
        motion_vectors: Array of motion vectors (dy, dx) for each block
        block_positions: Array of block center positions
        sad_values: Array of SAD values for matched blocks
    """
    h, w = frame1.shape
    motion_vectors = []
    block_positions = []
    sad_values = []

    print(f"\nComputing block matching with block size: {block_size}x{block_size}")
    print(f"Search range: ±{search_range} pixels")

    start_time = time.time()
    block_count = 0

    # Iterate through blocks in frame1
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            # Extract block from frame1
            block1 = frame1[y:y+block_size, x:x+block_size]

            # Define search region in frame2
            search_y_min = max(0, y - search_range)
            search_y_max = min(h - block_size, y + search_range)
            search_x_min = max(0, x - search_range)
            search_x_max = min(w - block_size, x + search_range)

            min_sad = float('inf')
            best_dy = 0
            best_dx = 0

            # Search for best matching block in frame2
            for search_y in range(search_y_min, search_y_max + 1):
                for search_x in range(search_x_min, search_x_max + 1):
                    block2 = frame2[search_y:search_y+block_size, search_x:search_x+block_size]

                    # Compute SAD
                    sad = compute_sad(block1, block2)

                    if sad < min_sad:
                        min_sad = sad
                        best_dy = search_y - y
                        best_dx = search_x - x

            # Store results
            motion_vectors.append([best_dx, best_dy])
            block_positions.append([x + block_size//2, y + block_size//2])
            sad_values.append(min_sad)
            block_count += 1

    elapsed_time = time.time() - start_time
    print(f"Processed {block_count} blocks in {elapsed_time:.2f} seconds")

    return np.array(motion_vectors), np.array(block_positions), np.array(sad_values)

# Compute motion vectors with 16x16 blocks
print("\n" + "="*80)
print("BLOCK MATCHING WITH 16x16 BLOCKS")
print("="*80)

motion_vectors_16, block_positions_16, sad_values_16 = block_matching_sad(
    frame1, frame2, block_size=16, search_range=32
)

# Calculate statistics
magnitudes_16 = np.sqrt(motion_vectors_16[:, 0]**2 + motion_vectors_16[:, 1]**2)

print(f"\nMotion Vector Statistics (16x16):")
print(f"  Total blocks: {len(motion_vectors_16)}")
print(f"  Magnitude range: [{magnitudes_16.min():.2f}, {magnitudes_16.max():.2f}] pixels")
print(f"  Mean magnitude: {magnitudes_16.mean():.2f} pixels")
print(f"  Median magnitude: {np.median(magnitudes_16):.2f} pixels")
print(f"  Std deviation: {magnitudes_16.std():.2f} pixels")

# Visualize motion vectors
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Plot 1: Motion vectors as arrows
frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB)
axes[0].imshow(frame1_rgb)
axes[0].set_title('Block Matching Motion Vectors (16x16 blocks, SAD)', fontsize=14)
axes[0].axis('off')

# Draw arrows for each block
for i in range(len(motion_vectors_16)):
    x, y = block_positions_16[i]
    dx, dy = motion_vectors_16[i]
    mag = magnitudes_16[i]

    # Color based on magnitude
    color = plt.cm.jet(mag / magnitudes_16.max()) if magnitudes_16.max() > 0 else 'blue'

    # Only draw arrow if there's significant motion
    if mag > 0.5:
        axes[0].arrow(x, y, dx, dy,
                     color=color,
                     head_width=4,
                     head_length=4,
                     linewidth=2,
                     alpha=0.7)

# Plot 2: Magnitude heatmap
magnitude_map = np.zeros_like(frame1, dtype=np.float32)
for i in range(len(motion_vectors_16)):
    x, y = block_positions_16[i]
    mag = magnitudes_16[i]
    # Fill the block with magnitude value
    y_start = max(0, y - 8)
    y_end = min(frame1.shape[0], y + 8)
    x_start = max(0, x - 8)
    x_end = min(frame1.shape[1], x + 8)
    magnitude_map[y_start:y_end, x_start:x_end] = mag

axes[1].imshow(frame1, cmap='gray', alpha=0.5)
im = axes[1].imshow(magnitude_map, cmap='hot', alpha=0.6)
axes[1].set_title('Motion Magnitude Heatmap', fontsize=14)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046, label='Magnitude (pixels)')

plt.tight_layout()
plt.savefig(output_folder / '02_motion_vectors_16x16.png', dpi=150, bbox_inches='tight')
print(f"Saved: 02_motion_vectors_16x16.png")
plt.show()

# Analyze different regions
print("\n" + "="*80)
print("REGION ANALYSIS")
print("="*80)

# Define thresholds
percentile_75 = np.percentile(magnitudes_16, 75)
percentile_25 = np.percentile(magnitudes_16, 25)

print(f"\nMagnitude Percentiles:")
print(f"  25th percentile: {percentile_25:.2f} pixels")
print(f"  75th percentile: {percentile_75:.2f} pixels")

# Categorize blocks
stationary_mask = magnitudes_16 < percentile_25
medium_motion_mask = (magnitudes_16 >= percentile_25) & (magnitudes_16 < percentile_75)
large_motion_mask = magnitudes_16 >= percentile_75

print(f"\nRegion Distribution:")
print(f"  Stationary regions: {np.sum(stationary_mask)} blocks ({np.sum(stationary_mask) / len(magnitudes_16) * 100:.1f}%)")
print(f"  Medium motion regions: {np.sum(medium_motion_mask)} blocks ({np.sum(medium_motion_mask) / len(magnitudes_16) * 100:.1f}%)")
print(f"  Large motion regions: {np.sum(large_motion_mask)} blocks ({np.sum(large_motion_mask) / len(magnitudes_16) * 100:.1f}%)")

# Visualize different regions
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Original frame
axes[0, 0].imshow(frame1, cmap='gray')
axes[0, 0].set_title('Original Frame', fontsize=14)
axes[0, 0].axis('off')

# All regions colored by motion
axes[0, 1].imshow(frame1, cmap='gray')
axes[0, 1].set_title('Motion Regions Classification', fontsize=14)
axes[0, 1].axis('off')

for i in range(len(motion_vectors_16)):
    x, y = block_positions_16[i]
    if stationary_mask[i]:
        color = 'blue'
        label = 'Stationary'
    elif large_motion_mask[i]:
        color = 'red'
        label = 'Large Motion'
    else:
        color = 'yellow'
        label = 'Medium Motion'

    # Draw a small square to indicate the block
    rect = plt.Rectangle((x-8, y-8), 16, 16, linewidth=1,
                         edgecolor=color, facecolor=color, alpha=0.3)
    axes[0, 1].add_patch(rect)

# Large motion regions only
axes[1, 0].imshow(frame1_rgb)
axes[1, 0].set_title('Large Motion Regions (Red Arrows)', fontsize=14)
axes[1, 0].axis('off')

for i in range(len(motion_vectors_16)):
    if large_motion_mask[i]:
        x, y = block_positions_16[i]
        dx, dy = motion_vectors_16[i]
        axes[1, 0].arrow(x, y, dx, dy, color='red',
                        head_width=4, head_length=4, linewidth=2.5, alpha=0.8)

# Stationary regions only
axes[1, 1].imshow(frame1_rgb)
axes[1, 1].set_title('Stationary Regions (Blue Arrows)', fontsize=14)
axes[1, 1].axis('off')

for i in range(len(motion_vectors_16)):
    if stationary_mask[i]:
        x, y = block_positions_16[i]
        dx, dy = motion_vectors_16[i]
        # Draw even small vectors to show they're nearly zero
        axes[1, 1].arrow(x, y, dx, dy, color='blue',
                        head_width=4, head_length=4, linewidth=2, alpha=0.8)

plt.tight_layout()
plt.savefig(output_folder / '03_region_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: 03_region_analysis.png")
plt.show()

# Test different block sizes
print("\n" + "="*80)
print("TESTING DIFFERENT BLOCK SIZES")
print("="*80)

block_sizes = [8, 16, 32]
results = {}

for block_size in block_sizes:
    print(f"\n--- Block Size: {block_size}x{block_size} ---")
    mv, bp, sad = block_matching_sad(frame1, frame2, block_size=block_size, search_range=32)
    mag = np.sqrt(mv[:, 0]**2 + mv[:, 1]**2)
    results[block_size] = {
        'motion_vectors': mv,
        'block_positions': bp,
        'sad_values': sad,
        'magnitudes': mag
    }
    print(f"  Number of blocks: {len(mv)}")
    print(f"  Mean magnitude: {mag.mean():.2f} pixels")
    print(f"  Max magnitude: {mag.max():.2f} pixels")

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

for idx, block_size in enumerate(block_sizes):
    frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB)
    axes[idx].imshow(frame_rgb)
    axes[idx].set_title(f'Block Size: {block_size}x{block_size}', fontsize=14)
    axes[idx].axis('off')

    mv = results[block_size]['motion_vectors']
    bp = results[block_size]['block_positions']
    mag = results[block_size]['magnitudes']

    max_mag = mag.max() if mag.max() > 0 else 1

    for i in range(len(mv)):
        x, y = bp[i]
        dx, dy = mv[i]
        m = mag[i]

        if m > 0.5:
            color = plt.cm.jet(m / max_mag)
            axes[idx].arrow(x, y, dx, dy,
                          color=color,
                          head_width=block_size//4,
                          head_length=block_size//4,
                          linewidth=2,
                          alpha=0.7)

plt.tight_layout()
plt.savefig(output_folder / '04_block_size_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: 04_block_size_comparison.png")
plt.show()

# Print comprehensive analysis
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS AND COMMENTARY")
print("="*80)

print("\n(a) REGIONS WITH LARGER MOTION VECTORS:")
print("-" * 80)
print("""
1. MOVING VEHICLES/OBJECTS:
   - Largest motion vectors appear in regions with moving cars/vehicles
   - These objects have significant displacement between frames
   - Motion vectors can reach up to {:.2f} pixels
   - Color-coded as RED in the visualization

2. FOREGROUND/NEARBY OBJECTS:
   - Objects closer to the camera exhibit larger apparent motion
   - Due to perspective: same real-world motion appears larger when closer
   - More pronounced displacement in pixel space

3. EDGES OF MOVING OBJECTS:
   - Sharp transitions between moving and static regions
   - Block matching finds strong directional matches
   - High confidence in motion direction
""".format(magnitudes_16.max()))

print("\n(b) STATIONARY REGIONS APPEARANCE:")
print("-" * 80)
print("""
1. CHARACTERISTICS:
   - Motion vectors are near-zero (< {:.2f} pixels on average)
   - Arrows are very short or invisible at visualization scale
   - Color-coded as BLUE in the visualization
   - Represent {:.1f}% of all blocks

2. TYPICAL STATIONARY REGIONS:
   - Sky and distant background
   - Road surfaces far from camera
   - Static buildings and infrastructure
   - Non-moving parts of the scene

3. BLOCK MATCHING BEHAVIOR:
   - Best match is typically at or very near the original position
   - SAD values are low (blocks are nearly identical)
   - Small motion vectors may be due to noise or compression artifacts
""".format(percentile_25, np.sum(stationary_mask) / len(magnitudes_16) * 100))

print("\n(c) EFFECT OF BLOCK SIZE ON ACCURACY:")
print("-" * 80)
print("""
BLOCK SIZE: 8x8 pixels
  Advantages:
    + Higher spatial resolution ({} blocks)
    + Better detail in motion field
    + Can capture small object movements
    + Better for scenes with fine details
  Disadvantages:
    - More susceptible to noise and texture variations
    - Computational cost: More blocks to process
    - May miss larger coherent motions
    - Aperture problem: ambiguous matches in uniform regions

BLOCK SIZE: 16x16 pixels (USED IN THIS ANALYSIS)
  Advantages:
    + Good balance between detail and robustness ({} blocks)
    + Less affected by noise
    + Sufficient texture for reliable matching
    + Standard choice in many applications
  Disadvantages:
    - May average out motion of small objects
    - Lower spatial resolution than 8x8

BLOCK SIZE: 32x32 pixels
  Advantages:
    + Most robust to noise ({} blocks)
    + Faster computation (fewer blocks)
    + Good for large uniform motions
    + Strong matching confidence
  Disadvantages:
    - Very low spatial resolution
    - Cannot capture small object details
    - May mix motions of different objects in same block
    - Poor for scenes with complex motion patterns

RECOMMENDATION:
  - Use 8x8 for high-detail scenes with small objects
  - Use 16x16 for general-purpose motion estimation (BEST BALANCE)
  - Use 32x32 for fast motion, simple scenes, or when speed is critical
""".format(
    len(results[8]['motion_vectors']),
    len(results[16]['motion_vectors']),
    len(results[32]['motion_vectors'])
))

print("\n" + "="*80)
print("All visualizations saved successfully!")
print(f"Output folder: {output_folder}")
print("="*80)
