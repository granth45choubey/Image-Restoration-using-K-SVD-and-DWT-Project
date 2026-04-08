"""
utils.py — Utility Functions for Image Restoration Project
===========================================================
Helper functions for image loading, preprocessing, and
noise generation. Follows the Gaussian noise model from
Gonzalez & Woods, "Digital Image Processing" (Chapter 5).

Noise Model:
    g(x, y) = f(x, y) + η(x, y)

    where:
        f(x, y) = original image
        η(x, y) ~ N(0, σ²) = Gaussian noise with zero mean
        g(x, y) = noisy (degraded) image
"""

import numpy as np
from PIL import Image
import io


def load_image_as_grayscale(uploaded_file) -> np.ndarray:
    """
    Load an uploaded image file and convert to grayscale.

    Converts to 8-bit grayscale (0–255) as used in standard
    image processing texts.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object.

    Returns
    -------
    np.ndarray
        Grayscale image as 2D numpy array with dtype uint8.
    """
    image = Image.open(uploaded_file).convert("L")
    return np.array(image, dtype=np.uint8)


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to a grayscale image.

    Implements the additive noise model:
        g(x, y) = f(x, y) + η(x, y)
        η ~ N(0, σ²)

    The noisy image is clipped to [0, 255] to maintain valid
    pixel range for 8-bit images.

    Parameters
    ----------
    image : np.ndarray
        Original grayscale image (dtype uint8, values 0–255).
    sigma : float
        Standard deviation of Gaussian noise.
        sigma = 0  → no noise
        sigma = 25 → moderate noise
        sigma = 50 → heavy noise
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy image (dtype uint8, values clipped to [0, 255]).
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    noisy_image = image.astype(np.float64) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] float range.

    Parameters
    ----------
    image : np.ndarray
        Input image with any value range.

    Returns
    -------
    np.ndarray
        Image normalized to [0.0, 1.0].
    """
    img = image.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min == 0:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 format [0, 255].

    Handles both float [0, 1] and arbitrary range inputs.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Image in uint8 format.
    """
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.0:
        return (image * 255.0).clip(0, 255).astype(np.uint8)
    return image.clip(0, 255).astype(np.uint8)


def image_to_bytes(image: np.ndarray) -> bytes:
    """
    Convert a grayscale numpy image to PNG bytes for downloading.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8).

    Returns
    -------
    bytes
        PNG-encoded image bytes.
    """
    img = Image.fromarray(image.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def resize_if_large(image: np.ndarray, max_dim: int = 256) -> np.ndarray:
    """
    Resize image if either dimension exceeds max_dim.

    This helps keep K-SVD computation time reasonable
    for educational demonstrations.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image.
    max_dim : int
        Maximum allowed dimension (default: 256).

    Returns
    -------
    np.ndarray
        Resized image (or original if already small enough).
    """
    h, w = image.shape
    if h <= max_dim and w <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_pil = Image.fromarray(image)
    img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(img_resized, dtype=np.uint8)
