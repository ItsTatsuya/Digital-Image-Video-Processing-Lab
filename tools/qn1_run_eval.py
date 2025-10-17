import cv2
import numpy as np
from scipy import ndimage
import os

image_path = os.path.join(os.path.dirname(__file__), '..', 'degraded_image.png')
image_path = os.path.abspath(image_path)

original = cv2.imread(image_path)
if original is None:
    raise FileNotFoundError(f'Image not found: {image_path}')
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float64)


def inverse_filter(image, degradation_kernel):
    # compute FFT of kernel with same shape as image
    H = np.fft.fft2(degradation_kernel, s=image.shape)
    G = np.fft.fft2(image)

    H_inv = np.zeros_like(H, dtype=complex)
    threshold = 1e-3 * np.max(np.abs(H))
    mask = np.abs(H) > threshold
    H_inv[mask] = 1 / H[mask]

    F_restored = G * H_inv
    restored = np.fft.ifft2(F_restored).real
    restored = np.clip(restored, 0, 255)

    return restored


def wiener_filter_search(image, degradation_kernel, k_values=None):
    """Perform Wiener filtering with a set of regularization constants K and
    return the restored image with the best PSNR (lowest MSE) compared to
    the original image passed via closure or outer scope.
    """
    if k_values is None:
        k_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    G = np.fft.fft2(image)
    H = np.fft.fft2(degradation_kernel, s=image.shape)
    H_conj = np.conj(H)

    best = None
    best_mse = float('inf')
    best_psnr = -float('inf')
    best_img = None

    for K in k_values:
        W = H_conj / (H * H_conj + K)
        F_est = W * G
        img_est = np.fft.ifft2(F_est).real
        img_est = np.clip(img_est, 0, 255)

        # compute MSE against available original in outer scope if present
        try:
            mse = np.mean((original_gray - img_est) ** 2)
        except Exception:
            mse = np.mean((image - img_est) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')
            best_img = img_est
            best = K

    return best_img, best, best_mse, best_psnr


degradation_kernel = np.ones((5, 5)) / 25

degraded = ndimage.convolve(original_gray, degradation_kernel)
degraded = np.clip(degraded, 0, 255)

restored = inverse_filter(degraded, degradation_kernel)

# compute MSE and PSNR
mse = np.mean((original_gray - restored) ** 2)
psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')
print(f'MSE: {mse:.4f}, PSNR: {psnr:.4f} dB')

# Save restored image for inspection
out_path = os.path.join(os.path.dirname(__file__), '..', 'Set-3', 'Question-1', 'restored_eval.png')
cv2.imwrite(out_path, restored.astype(np.uint8))
print(f'Restored image written to {out_path}')

# Try Wiener filter search to improve restoration
best_img, best_k, best_mse, best_psnr = wiener_filter_search(degraded, degradation_kernel)
print(f'Wiener best K: {best_k}, MSE: {best_mse:.4f}, PSNR: {best_psnr:.4f} dB')
out_path_w = os.path.join(os.path.dirname(__file__), '..', 'Set-3', 'Question-1', 'restored_wiener_eval.png')
cv2.imwrite(out_path_w, best_img.astype(np.uint8))
print(f'Wiener restored image written to {out_path_w}')
