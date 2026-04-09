"""
metrics.py — Image Quality Assessment Metrics
==============================================
Implements standard image quality metrics used in image restoration
research, following conventions from Gonzalez & Woods,
"Digital Image Processing."

Metrics:
    - PSNR (Peak Signal-to-Noise Ratio)
    - MSE  (Mean Squared Error)
    - SSIM (Structural Similarity Index)
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_mse(original: np.ndarray, restored: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE) between two images.

    MSE = (1 / M*N) * Σ [f(x,y) - g(x,y)]²

    where f is the original image and g is the restored image.

    Parameters
    ----------
    original : np.ndarray
        Original (ground truth) image, values in [0, 255].
    restored : np.ndarray
        Restored / noisy image, values in [0, 255].

    Returns
    -------
    float
        MSE value. Lower is better (0 = identical).
    """
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    mse_value = np.mean((original - restored) ** 2)
    return float(mse_value)


def compute_psnr(original: np.ndarray, restored: np.ndarray, peak: float = 255.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(peak² / MSE)

    A higher PSNR indicates the restored image is closer to the original.
    Typical values for 8-bit images:
        - 30+ dB  → Good quality
        - 20–30 dB → Acceptable
        - <20 dB  → Poor quality

    Parameters
    ----------
    original : np.ndarray
        Original image, values in [0, 255].
    restored : np.ndarray
        Restored image, values in [0, 255].
    peak : float
        Maximum possible pixel value (255 for 8-bit images).

    Returns
    -------
    float
        PSNR in decibels (dB). Higher is better.
    """
    mse_value = compute_mse(original, restored)
    if mse_value == 0:
        return float('inf')  # Identical images
    psnr_value = 10.0 * np.log10((peak ** 2) / mse_value)
    return float(psnr_value)


def compute_ssim(original: np.ndarray, restored: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    SSIM measures perceived quality by comparing luminance,
    contrast, and structure between two images.

    SSIM ∈ [-1, 1], where 1 = perfectly identical.

    Parameters
    ----------
    original : np.ndarray
        Original image (grayscale), values in [0, 255].
    restored : np.ndarray
        Restored image (grayscale), values in [0, 255].

    Returns
    -------
    float
        SSIM value. Higher is better (1.0 = identical).
    """
    if np.array_equal(original, restored):
        return 1.0
        
    # Determine data_range from input dtype
    data_range = 255.0 if original.max() > 1.0 else 1.0
    ssim_value = ssim(original, restored, data_range=data_range)
    return float(ssim_value)


def compute_all_metrics(original: np.ndarray, restored: np.ndarray) -> dict:
    """
    Compute all quality metrics at once.

    Parameters
    ----------
    original : np.ndarray
        Original image.
    restored : np.ndarray
        Restored image.

    Returns
    -------
    dict
        Dictionary with keys 'MSE', 'PSNR', 'SSIM'.
    """
    return {
        "MSE": compute_mse(original, restored),
        "PSNR": compute_psnr(original, restored),
        "SSIM": compute_ssim(original, restored),
    }


def compute_difference_map(original: np.ndarray, restored: np.ndarray,
                           amplify: float = 3.0) -> np.ndarray:
    """
    Compute absolute difference map between original and restored images.

    |f(x,y) - g(x,y)| shows WHERE restoration errors occur.
    Bright regions = large error, dark regions = small error.

    The difference is amplified for visibility since raw differences
    tend to be small and hard to see.

    Parameters
    ----------
    original : np.ndarray
        Original image.
    restored : np.ndarray
        Restored image.
    amplify : float
        Amplification factor for visibility (default: 3.0).

    Returns
    -------
    np.ndarray
        Difference map as uint8 image (amplified).
    """
    diff = np.abs(original.astype(np.float64) - restored.astype(np.float64))
    diff = diff * amplify
    return np.clip(diff, 0, 255).astype(np.uint8)
