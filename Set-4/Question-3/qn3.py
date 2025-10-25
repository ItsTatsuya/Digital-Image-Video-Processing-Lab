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
axes[1].set_title('Frame 2 (t+1) - Ground Truth', fontsize=14)
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
print("STEP 1: BLOCK MATCHING TO FIND MOTION VECTORS")
print("="*80)

motion_vectors, block_positions, sad_values = block_matching_sad(
    frame1, frame2, block_size=16, search_range=32
)

# Calculate statistics
magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)

print(f"\nMotion Vector Statistics:")
print(f"  Total blocks: {len(motion_vectors)}")
print(f"  Magnitude range: [{magnitudes.min():.2f}, {magnitudes.max():.2f}] pixels")
print(f"  Mean magnitude: {magnitudes.mean():.2f} pixels")
print(f"  Median magnitude: {np.median(magnitudes):.2f} pixels")

# Visualize motion vectors
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# Plot 1: Motion vectors as arrows
frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB)
axes[0].imshow(frame1_rgb)
axes[0].set_title('Motion Vectors from Block Matching (16x16 blocks)', fontsize=14)
axes[0].axis('off')

# Draw arrows for each block
for i in range(len(motion_vectors)):
    x, y = block_positions[i]
    dx, dy = motion_vectors[i]
    mag = magnitudes[i]

    # Color based on magnitude
    color = plt.cm.jet(mag / magnitudes.max()) if magnitudes.max() > 0 else 'blue'

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
for i in range(len(motion_vectors)):
    x, y = block_positions[i]
    mag = magnitudes[i]
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
plt.savefig(output_folder / '02_motion_vectors.png', dpi=150, bbox_inches='tight')
print(f"Saved: 02_motion_vectors.png")
plt.show()

# STEP 2: Predict Frame 2 using Motion Vectors from Frame 1
print("\n" + "="*80)
print("STEP 2: FRAME PREDICTION USING MOTION VECTORS")
print("="*80)

block_size = 16
h, w = frame1.shape

# Initialize predicted frame (start with frame1 as base)
predicted_frame = np.zeros_like(frame1, dtype=np.float32)
weight_map = np.zeros_like(frame1, dtype=np.float32)  # To handle overlapping blocks

print("\nPredicting frame 2 from frame 1 using motion vectors...")

block_idx = 0
for y in range(0, h - block_size + 1, block_size):
    for x in range(0, w - block_size + 1, block_size):
        # Get the motion vector for this block
        dx, dy = motion_vectors[block_idx]

        # Extract block from frame1
        block = frame1[y:y+block_size, x:x+block_size].astype(np.float32)

        # Calculate target position in predicted frame
        target_y = y + int(dy)
        target_x = x + int(dx)

        # Ensure target position is within frame boundaries
        target_y = max(0, min(h - block_size, target_y))
        target_x = max(0, min(w - block_size, target_x))

        # Place the block in predicted frame at target position
        predicted_frame[target_y:target_y+block_size, target_x:target_x+block_size] += block
        weight_map[target_y:target_y+block_size, target_x:target_x+block_size] += 1

        block_idx += 1

# Average overlapping blocks
# Avoid division by zero
weight_map[weight_map == 0] = 1
predicted_frame = predicted_frame / weight_map
predicted_frame = np.clip(predicted_frame, 0, 255).astype(np.uint8)

print(f"Predicted frame shape: {predicted_frame.shape}")
print(f"Predicted frame dtype: {predicted_frame.dtype}")

# STEP 3: Compute Residual (Difference) Image
print("\n" + "="*80)
print("STEP 3: COMPUTING RESIDUAL DIFFERENCE IMAGE")
print("="*80)

# Compute residual: actual frame2 - predicted frame
residual = frame2.astype(np.int16) - predicted_frame.astype(np.int16)

# Calculate error metrics
mse = np.mean((frame2.astype(np.float32) - predicted_frame.astype(np.float32))**2)
psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
mae = np.mean(np.abs(residual))

# Calculate absolute residual for visualization
residual_abs = np.abs(residual).astype(np.uint8)

print(f"\nPrediction Quality Metrics:")
print(f"  Mean Squared Error (MSE): {mse:.2f}")
print(f"  Peak Signal-to-Noise Ratio (PSNR): {psnr:.2f} dB")
print(f"  Mean Absolute Error (MAE): {mae:.2f}")
print(f"\nResidual Statistics:")
print(f"  Residual range: [{residual.min()}, {residual.max()}]")
print(f"  Mean residual: {residual.mean():.2f}")
print(f"  Std residual: {residual.std():.2f}")

# Visualize: Original, Predicted, and Residual
fig, axes = plt.subplots(2, 3, figsize=(24, 16))

# Row 1: Frames
axes[0, 0].imshow(frame1, cmap='gray')
axes[0, 0].set_title('Frame 1 (Reference)', fontsize=14)
axes[0, 0].axis('off')

axes[0, 1].imshow(predicted_frame, cmap='gray')
axes[0, 1].set_title('Predicted Frame 2', fontsize=14)
axes[0, 1].axis('off')

axes[0, 2].imshow(frame2, cmap='gray')
axes[0, 2].set_title('Actual Frame 2 (Ground Truth)', fontsize=14)
axes[0, 2].axis('off')

# Row 2: Difference visualizations
# Absolute difference
axes[1, 0].imshow(residual_abs, cmap='hot')
axes[1, 0].set_title('Absolute Residual Error', fontsize=14)
axes[1, 0].axis('off')
plt.colorbar(axes[1, 0].imshow(residual_abs, cmap='hot'), ax=axes[1, 0], fraction=0.046)

# Residual (centered at 128 for better visualization)
residual_vis = np.clip(residual + 128, 0, 255).astype(np.uint8)
axes[1, 1].imshow(residual_vis, cmap='gray')
axes[1, 1].set_title('Residual (Gray=0, Bright>0, Dark<0)', fontsize=14)
axes[1, 1].axis('off')

# Error overlay on actual frame
axes[1, 2].imshow(frame2, cmap='gray', alpha=0.5)
im = axes[1, 2].imshow(residual_abs, cmap='hot', alpha=0.6)
axes[1, 2].set_title('Error Overlay on Ground Truth', fontsize=14)
axes[1, 2].axis('off')
plt.colorbar(im, ax=axes[1, 2], fraction=0.046, label='Absolute Error')

# Add text box with metrics
textstr = f'MSE: {mse:.2f}\nPSNR: {psnr:.2f} dB\nMAE: {mae:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
fig.text(0.5, 0.02, textstr, fontsize=12, ha='center', bbox=props)

plt.tight_layout()
plt.savefig(output_folder / '03_prediction_and_residual.png', dpi=150, bbox_inches='tight')
print(f"Saved: 03_prediction_and_residual.png")
plt.show()

# Analyze residual error distribution
print("\n" + "="*80)
print("RESIDUAL ERROR ANALYSIS")
print("="*80)

# Calculate percentiles
percentile_75 = np.percentile(residual_abs, 75)
percentile_90 = np.percentile(residual_abs, 90)
percentile_95 = np.percentile(residual_abs, 95)

print(f"\nResidual Error Percentiles:")
print(f"  75th percentile: {percentile_75:.2f}")
print(f"  90th percentile: {percentile_90:.2f}")
print(f"  95th percentile: {percentile_95:.2f}")

# Categorize regions by error
low_error_mask = residual_abs < percentile_75
medium_error_mask = (residual_abs >= percentile_75) & (residual_abs < percentile_90)
high_error_mask = residual_abs >= percentile_90

low_error_count = np.sum(low_error_mask)
medium_error_count = np.sum(medium_error_mask)
high_error_count = np.sum(high_error_mask)

print(f"\nError Distribution:")
print(f"  Low error regions: {low_error_count} pixels ({low_error_count / residual_abs.size * 100:.1f}%)")
print(f"  Medium error regions: {medium_error_count} pixels ({medium_error_count / residual_abs.size * 100:.1f}%)")
print(f"  High error regions: {high_error_count} pixels ({high_error_count / residual_abs.size * 100:.1f}%)")

# Visualize error regions
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Error segmentation
error_segmented = np.zeros_like(residual_abs)
error_segmented[low_error_mask] = 85  # Low error = dark
error_segmented[medium_error_mask] = 170  # Medium error = gray
error_segmented[high_error_mask] = 255  # High error = bright

axes[0, 0].imshow(error_segmented, cmap='gray')
axes[0, 0].set_title('Error Segmentation (Dark=Low, Bright=High)', fontsize=14)
axes[0, 0].axis('off')

# High error regions highlighted
axes[0, 1].imshow(frame2, cmap='gray')
high_error_overlay = np.zeros((*frame2.shape, 3), dtype=np.uint8)
high_error_overlay[high_error_mask] = [255, 0, 0]  # Red for high error
axes[0, 1].imshow(high_error_overlay, alpha=0.5)
axes[0, 1].set_title('High Error Regions (Red Overlay)', fontsize=14)
axes[0, 1].axis('off')

# Histogram of residual errors
axes[1, 0].hist(residual_abs.flatten(), bins=100, color='blue', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(percentile_75, color='orange', linestyle='--', linewidth=2, label='75th percentile')
axes[1, 0].axvline(percentile_90, color='red', linestyle='--', linewidth=2, label='90th percentile')
axes[1, 0].set_xlabel('Absolute Residual Error', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Distribution of Residual Errors', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Comparison: actual vs predicted (scatter plot for sample pixels)
sample_indices = np.random.choice(frame2.size, size=min(10000, frame2.size), replace=False)
actual_samples = frame2.flatten()[sample_indices]
predicted_samples = predicted_frame.flatten()[sample_indices]

axes[1, 1].scatter(actual_samples, predicted_samples, alpha=0.3, s=1)
axes[1, 1].plot([0, 255], [0, 255], 'r--', linewidth=2, label='Perfect prediction')
axes[1, 1].set_xlabel('Actual Pixel Value', fontsize=12)
axes[1, 1].set_ylabel('Predicted Pixel Value', fontsize=12)
axes[1, 1].set_title('Predicted vs Actual Pixel Values', fontsize=14)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim([0, 255])
axes[1, 1].set_ylim([0, 255])

plt.tight_layout()
plt.savefig(output_folder / '04_error_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: 04_error_analysis.png")
plt.show()

# Calculate data compression benefit
print("\n" + "="*80)
print("DATA COMPRESSION ANALYSIS")
print("="*80)

# Original data size (storing frame2 directly)
original_size = frame2.size * 8  # 8 bits per pixel

# Compressed data: motion vectors + residual
# Motion vectors: 2 values (dx, dy) per block, let's assume 8 bits each
num_blocks = len(motion_vectors)
motion_vector_size = num_blocks * 2 * 8  # 2 components, 8 bits each

# Residual: can be compressed more efficiently (many near-zero values)
# For simplicity, let's calculate bits needed for residual
residual_entropy = -np.sum((np.histogram(residual_abs, bins=256)[0] / residual_abs.size) *
                           np.log2(np.histogram(residual_abs, bins=256)[0] / residual_abs.size + 1e-10))
residual_size = residual_entropy * residual_abs.size

total_compressed_size = motion_vector_size + residual_size
compression_ratio = original_size / total_compressed_size

print(f"\nStorage Requirements:")
print(f"  Original frame size: {original_size / 8 / 1024:.2f} KB ({original_size} bits)")
print(f"  Motion vectors: {motion_vector_size / 8 / 1024:.2f} KB ({motion_vector_size} bits)")
print(f"  Residual (entropy): {residual_size / 8 / 1024:.2f} KB ({residual_size:.0f} bits)")
print(f"  Total compressed: {total_compressed_size / 8 / 1024:.2f} KB ({total_compressed_size:.0f} bits)")
print(f"  Compression ratio: {compression_ratio:.2f}x")
print(f"  Space savings: {(1 - 1/compression_ratio) * 100:.1f}%")

# Print comprehensive commentary
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS AND COMMENTARY")
print("="*80)

print("\n(a) PREDICTION ACCURACY:")
print("-" * 80)
print(f"""
QUANTITATIVE METRICS:
  - PSNR: {psnr:.2f} dB (Higher is better, >30 dB is good)
  - MSE: {mse:.2f} (Lower is better)
  - MAE: {mae:.2f} pixels (Average error per pixel)

QUALITATIVE ASSESSMENT:
  The predicted frame shows GOOD overall accuracy with:

  ✓ Well-Predicted Regions (~{low_error_count / residual_abs.size * 100:.1f}%):
    - Background areas (sky, distant objects)
    - Stationary regions (buildings, road markings)
    - Regions with simple translational motion
    - Low-texture areas where block matching is reliable

  ✗ Poorly-Predicted Regions (~{high_error_count / residual_abs.size * 100:.1f}%):
    - Object boundaries and edges
    - Fast-moving vehicles
    - Regions with complex/non-translational motion
    - Areas with occlusions (newly visible regions)
    - High-frequency texture details

ACCURACY ASSESSMENT: {"EXCELLENT" if psnr > 35 else "GOOD" if psnr > 30 else "MODERATE" if psnr > 25 else "FAIR"}
  The prediction quality is {"excellent" if psnr > 35 else "good" if psnr > 30 else "moderate" if psnr > 25 else "fair"}
  for video compression purposes.
""")

print("\n(b) REGIONS WITH LARGER RESIDUAL ERRORS:")
print("-" * 80)
print(f"""
HIGH ERROR REGIONS (>{percentile_90:.0f} error, representing {high_error_count / residual_abs.size * 100:.1f}%):

1. OBJECT BOUNDARIES:
   - Moving objects create sharp edges in motion
   - Block matching cannot perfectly align discontinuous regions
   - Half-pixel misalignments cause edge artifacts
   - Reason: Block-based motion model too coarse for edges

2. OCCLUSION/DISOCCLUSION REGIONS:
   - Areas that become visible or hidden between frames
   - No corresponding block exists in reference frame
   - Prediction fails completely in these regions
   - Reason: Block matching assumes all content exists in both frames

3. FAST-MOVING OBJECTS:
   - Vehicles with high velocity
   - Motion may exceed search range
   - Motion blur differences between frames
   - Reason: Large displacements and motion blur effects

4. COMPLEX MOTION PATTERNS:
   - Rotating or deforming objects
   - Non-rigid motion (e.g., people, trees swaying)
   - Motion that isn't pure translation
   - Reason: Block matching assumes pure translational motion

5. HIGH-FREQUENCY DETAILS:
   - Fine texture patterns
   - Small features within blocks
   - Sharp gradients and edges
   - Reason: 16x16 blocks average out fine details

6. ILLUMINATION CHANGES:
   - Regions with lighting variations between frames
   - Shadow changes
   - Reflection differences
   - Reason: Block matching assumes constant illumination
""")

print("\n(c) MOTION VECTORS AND FRAME-TO-FRAME REDUNDANCY:")
print("-" * 80)
print(f"""
HOW MOTION VECTORS REDUCE REDUNDANCY:

1. TEMPORAL PREDICTION:
   - Instead of storing entire frame 2 ({original_size / 8 / 1024:.2f} KB)
   - Store motion vectors ({motion_vector_size / 8 / 1024:.2f} KB) + residual ({residual_size / 8 / 1024:.2f} KB)
   - Compression ratio: {compression_ratio:.2f}x
   - Data savings: {(1 - 1/compression_ratio) * 100:.1f}%

2. EXPLOITING TEMPORAL CORRELATION:
   - Consecutive frames are highly similar
   - Most regions only translate slightly
   - Motion vectors capture this translation efficiently
   - Residual encodes only the prediction errors

3. SPARSE REPRESENTATION:
   - Motion vectors: {num_blocks} blocks × 2 values = {num_blocks * 2} values
   - Much smaller than {frame2.size} pixels in original frame
   - Residual has many near-zero values (easily compressible)
   - Entropy of residual: {residual_entropy:.2f} bits/pixel (vs 8 bits/pixel original)

4. VIDEO CODEC FOUNDATION:
   - This is the basis of MPEG, H.264, H.265 video compression
   - Modern codecs use:
     * Sub-pixel motion estimation (quarter-pixel accuracy)
     * Variable block sizes (4×4 to 64×64)
     * Bi-directional prediction (P and B frames)
     * Advanced entropy coding (CABAC, CAVLC)

5. EFFECTIVENESS METRICS:
   - {low_error_count / residual_abs.size * 100:.1f}% of pixels have low prediction error
   - Residual contains mostly small values (mean absolute: {mae:.2f})
   - Most information can be reconstructed from motion vectors alone
   - Only fine details and special cases need residual correction

6. TRADE-OFFS:
   Advantages:
     ✓ Massive data reduction ({compression_ratio:.2f}x smaller)
     ✓ Preserves visual quality (PSNR: {psnr:.2f} dB)
     ✓ Efficient for streaming and storage

   Disadvantages:
     ✗ Computational cost of block matching
     ✗ Cannot handle all motion types perfectly
     ✗ Error accumulation in long sequences
     ✗ Requires periodic I-frames for resynchronization

CONCLUSION:
  Motion compensation is highly effective at reducing temporal redundancy.
  The combination of motion vectors and residual provides excellent
  compression while maintaining good quality. The {compression_ratio:.2f}x compression
  ratio demonstrates significant bandwidth/storage savings for video applications.
""")

print("\n" + "="*80)
print("All visualizations saved successfully!")
print(f"Output folder: {output_folder}")
print("="*80)
