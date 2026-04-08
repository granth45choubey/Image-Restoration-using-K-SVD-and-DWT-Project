"""
wavelet.py — DWT-Based Image Denoising
========================================
Implements image denoising using the Discrete Wavelet Transform (DWT)
with soft thresholding, following concepts from Gonzalez & Woods,
"Digital Image Processing" (Chapter 7 — Wavelets).

Theory:
    The DWT decomposes an image into four subbands:
        LL — Approximation (low-frequency content, main structure)
        LH — Horizontal detail (horizontal edges)
        HL — Vertical detail (vertical edges)
        HH — Diagonal detail (diagonal edges / noise)

    Noise primarily resides in the high-frequency detail subbands
    (LH, HL, HH). By applying soft thresholding to these subbands,
    we attenuate noise while preserving significant image features.

Soft Thresholding (Donoho & Johnstone):
    For each coefficient w:
        if |w| <= T:  w_hat = 0
        if |w| > T:   w_hat = sign(w) * (|w| - T)

    where T = α * max(|coefficients|)

    Alpha (α) controls thresholding strength:
        α → 0: minimal thresholding (preserves detail + noise)
        α → 1: aggressive thresholding (removes noise + some detail)

Reference: Gonzalez & Woods, "Digital Image Processing", Chapter 7
"""

import numpy as np
import pywt


def soft_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to wavelet coefficients.

    Implements the shrinkage function:
        w_hat = sign(w) * max(|w| - T, 0)

    This smoothly attenuates coefficients near zero (noise)
    while preserving large coefficients (signal).

    Parameters
    ----------
    coefficients : np.ndarray
        Wavelet detail coefficients (LH, HL, or HH subband).
    threshold : float
        Threshold value T. Coefficients with |w| <= T are set to zero.

    Returns
    -------
    np.ndarray
        Thresholded coefficients (same shape as input).
    """
    return np.sign(coefficients) * np.maximum(np.abs(coefficients) - threshold, 0.0)


def compute_threshold(coefficients: np.ndarray, alpha: float) -> float:
    """
    Compute the threshold value from alpha and coefficient magnitude.

    T = α * max(|coefficients|)

    Parameters
    ----------
    coefficients : np.ndarray
        Wavelet coefficients from a detail subband.
    alpha : float
        Thresholding strength in [0.1, 1.0].

    Returns
    -------
    float
        Computed threshold value.
    """
    max_abs = np.max(np.abs(coefficients))
    return alpha * max_abs


def dwt_denoise(image: np.ndarray, alpha: float, wavelet: str = "db1") -> dict:
    """
    Denoise an image using single-level DWT with soft thresholding.

    Pipeline:
        1. Convert image to float64 [0, 255]
        2. Apply 2D DWT → (LL, (LH, HL, HH))
        3. Compute threshold T = α * max(|detail coefficients|)
        4. Apply soft thresholding to LH, HL, HH subbands
        5. Reconstruct image using inverse DWT (IDWT)
        6. Clip result to [0, 255] and convert to uint8

    Parameters
    ----------
    image : np.ndarray
        Noisy grayscale image (uint8, values 0–255).
    alpha : float
        Thresholding strength in [0.1, 1.0].
        - Low alpha → mild denoising   (preserves detail)
        - High alpha → strong denoising (smooths image)
    wavelet : str
        Wavelet family to use. Default is 'db1' (Haar wavelet).
        'db1' / 'haar' — simplest, good for educational purposes.

    Returns
    -------
    dict
        Dictionary containing:
        - 'denoised'     : np.ndarray — Restored image (uint8)
        - 'LL'           : np.ndarray — Approximation subband
        - 'LH'           : np.ndarray — Horizontal detail (original)
        - 'HL'           : np.ndarray — Vertical detail (original)
        - 'HH'           : np.ndarray — Diagonal detail (original)
        - 'LH_thresh'    : np.ndarray — LH after thresholding
        - 'HL_thresh'    : np.ndarray — HL after thresholding
        - 'HH_thresh'    : np.ndarray — HH after thresholding
        - 'threshold'    : float      — Threshold value used
        - 'wavelet'      : str        — Wavelet name used
        - 'alpha'        : float      — Alpha value used
    """
    # Step 1: Convert to float for precision
    img_float = image.astype(np.float64)

    # Step 2: Apply 2D DWT decomposition
    # pywt.dwt2 returns: (LL, (LH, HL, HH))
    LL, (LH, HL, HH) = pywt.dwt2(img_float, wavelet)

    # Step 3: Compute threshold from all detail coefficients
    # We use the maximum absolute value across ALL detail subbands
    # for a unified threshold (Gonzalez approach)
    all_details = np.concatenate([LH.ravel(), HL.ravel(), HH.ravel()])
    threshold = compute_threshold(all_details, alpha)

    # Step 4: Apply soft thresholding to detail subbands ONLY
    # LL (approximation) is NOT thresholded — it contains the
    # main image structure and should be preserved
    LH_thresh = soft_threshold(LH, threshold)
    HL_thresh = soft_threshold(HL, threshold)
    HH_thresh = soft_threshold(HH, threshold)

    # Step 5: Reconstruct image using inverse DWT
    denoised = pywt.idwt2((LL, (LH_thresh, HL_thresh, HH_thresh)), wavelet)

    # Step 6: Handle size mismatch (DWT may add 1 pixel)
    denoised = denoised[:image.shape[0], :image.shape[1]]

    # Step 7: Clip to valid range and convert to uint8
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)

    return {
        "denoised": denoised,
        # Original subbands (before thresholding)
        "LL": LL,
        "LH": LH,
        "HL": HL,
        "HH": HH,
        # Thresholded subbands
        "LH_thresh": LH_thresh,
        "HL_thresh": HL_thresh,
        "HH_thresh": HH_thresh,
        # Parameters
        "threshold": threshold,
        "wavelet": wavelet,
        "alpha": alpha,
    }


def normalize_subband_for_display(subband: np.ndarray) -> np.ndarray:
    """
    Normalize a wavelet subband for visual display.

    Detail subbands (LH, HL, HH) contain positive and negative values.
    This function maps them to [0, 255] for visualization.

    Parameters
    ----------
    subband : np.ndarray
        Wavelet subband coefficients.

    Returns
    -------
    np.ndarray
        Normalized subband as uint8 for display.
    """
    sub = subband.astype(np.float64)
    sub_min, sub_max = sub.min(), sub.max()
    if sub_max - sub_min == 0:
        return np.zeros_like(sub, dtype=np.uint8)
    normalized = ((sub - sub_min) / (sub_max - sub_min) * 255.0)
    return normalized.astype(np.uint8)
