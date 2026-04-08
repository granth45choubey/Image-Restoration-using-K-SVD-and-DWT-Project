"""
ksvd.py — K-SVD Image Denoising (Implemented from Scratch)
============================================================
Implements the K-SVD algorithm for image denoising using sparse
representation over learned dictionaries, following concepts from
Gonzalez & Woods, "Digital Image Processing" and the original
K-SVD paper by Aharon, Elad & Bruckstein (2006).

Theory:
    K-SVD operates in the SPATIAL DOMAIN by:
    1. Extracting small overlapping patches from the image
    2. Learning a dictionary D such that each patch y ≈ D·x
       where x is a SPARSE coefficient vector
    3. Using the learned dictionary to reconstruct clean patches

    The key insight: natural images can be represented sparsely
    using a learned dictionary, but noise CANNOT. So by enforcing
    sparsity, we keep the signal and remove the noise.

Algorithm Components:
    - Patch Extraction: Overlapping patches → column vectors
    - Dictionary Init:  DCT-based initialization (Gonzalez, Ch. 4)
    - Sparse Coding:    OMP (Orthogonal Matching Pursuit)
    - Dictionary Update: SVD-based atom update (the "K-SVD" step)
    - Reconstruction:   Average overlapping patches

Alpha (α) controls sparsity:
    - Higher α → fewer non-zero coefficients → stronger denoising
    - Lower α  → more non-zero coefficients → preserves more detail
    - max_nonzero = max(1, round((1 - α) × patch_size))

Reference:
    Aharon, M., Elad, M., & Bruckstein, A. (2006).
    "K-SVD: An Algorithm for Designing Overcomplete Dictionaries
     for Sparse Representation."
"""

import numpy as np


# ═════════════════════════════════════════════════════════════
# STEP 1: Patch Extraction & Reconstruction
# ═════════════════════════════════════════════════════════════

def extract_patches(image, patch_size, step=None):
    """
    Extract overlapping patches from a grayscale image.

    Each patch is a (patch_size × patch_size) block extracted
    from the image, then flattened into a column vector.

    Patches overlap by (patch_size - step) pixels.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H × W), float64.
    patch_size : int
        Size of each square patch (e.g., 8 for 8×8 patches).
    step : int or None
        Step size between patches. Default = patch_size // 2
        (50% overlap, good balance of quality and speed).

    Returns
    -------
    Y : np.ndarray
        Patch matrix of shape (d × N), where:
        d = patch_size² (dimension of each patch vector)
        N = number of patches extracted
    positions : list of (int, int)
        Top-left (row, col) position of each patch in the image.
    """
    if step is None:
        step = max(1, patch_size // 2)

    H, W = image.shape
    d = patch_size * patch_size  # Dimension of each patch vector
    patches = []
    positions = []

    # Slide a window across the image
    for i in range(0, H - patch_size + 1, step):
        for j in range(0, W - patch_size + 1, step):
            # Extract patch and flatten to column vector
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch.ravel())
            positions.append((i, j))

    # Stack as (d × N) matrix — each column is one patch
    Y = np.array(patches).T
    return Y, positions


def reconstruct_from_patches(patch_matrix, positions, image_shape, patch_size):
    """
    Reconstruct an image from overlapping patches by averaging.

    Where patches overlap, we AVERAGE the pixel values. This
    produces smooth transitions and avoids blocking artifacts.

    Parameters
    ----------
    patch_matrix : np.ndarray
        Patch matrix (d × N), each column is a reconstructed patch.
    positions : list of (int, int)
        Top-left positions of each patch.
    image_shape : tuple
        (H, W) shape of the output image.
    patch_size : int
        Size of each square patch.

    Returns
    -------
    np.ndarray
        Reconstructed image (uint8, clipped to [0, 255]).
    """
    reconstructed = np.zeros(image_shape, dtype=np.float64)
    weight = np.zeros(image_shape, dtype=np.float64)

    for idx, (i, j) in enumerate(positions):
        # Reshape column vector back to 2D patch
        patch = patch_matrix[:, idx].reshape(patch_size, patch_size)

        # Accumulate patch values and count overlaps
        reconstructed[i:i + patch_size, j:j + patch_size] += patch
        weight[i:i + patch_size, j:j + patch_size] += 1.0

    # Average overlapping regions
    weight = np.maximum(weight, 1.0)  # Avoid division by zero
    reconstructed = reconstructed / weight

    return np.clip(reconstructed, 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════
# STEP 2: Dictionary Initialization (DCT-based)
# ═════════════════════════════════════════════════════════════

def create_dct_dictionary(patch_size, n_atoms=None):
    """
    Create an initial dictionary using 2D DCT basis functions.

    The Discrete Cosine Transform (DCT) provides a natural basis
    for image patches (Gonzalez, Ch. 4). Each atom corresponds to
    a specific 2D frequency pattern.

    For a patch_size of 8, this creates 64 DCT atoms covering
    all frequency combinations from DC (0,0) to the highest
    frequency (7,7).

    DCT basis function:
        D[n1, n2] = cos(π(2n1+1)k1 / 2N) · cos(π(2n2+1)k2 / 2N)

    Parameters
    ----------
    patch_size : int
        Size of the square patch (e.g., 8).
    n_atoms : int or None
        Number of dictionary atoms. Default = patch_size².
        If > patch_size², extra atoms are initialized randomly.

    Returns
    -------
    D : np.ndarray
        Dictionary matrix (d × n_atoms), each column is a
        normalized atom. d = patch_size².
    """
    if n_atoms is None:
        n_atoms = patch_size * patch_size

    d = patch_size * patch_size
    D = np.zeros((d, n_atoms))
    atom_count = 0

    # Generate 2D DCT basis atoms
    for k1 in range(patch_size):
        for k2 in range(patch_size):
            if atom_count >= n_atoms:
                break

            # Build the 2D DCT atom for frequency (k1, k2)
            atom = np.zeros((patch_size, patch_size))
            for n1 in range(patch_size):
                for n2 in range(patch_size):
                    atom[n1, n2] = (
                        np.cos(np.pi * (2 * n1 + 1) * k1 / (2 * patch_size))
                        * np.cos(np.pi * (2 * n2 + 1) * k2 / (2 * patch_size))
                    )

            # Normalize to unit length
            vec = atom.ravel()
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            D[:, atom_count] = vec
            atom_count += 1

        if atom_count >= n_atoms:
            break

    # Fill remaining atoms with random normalized vectors
    # (for overcomplete dictionaries where n_atoms > patch_size²)
    rng = np.random.default_rng(42)
    while atom_count < n_atoms:
        vec = rng.standard_normal(d)
        vec = vec / np.linalg.norm(vec)
        D[:, atom_count] = vec
        atom_count += 1

    return D


# ═════════════════════════════════════════════════════════════
# STEP 3: Sparse Coding — Orthogonal Matching Pursuit (OMP)
# ═════════════════════════════════════════════════════════════

def omp(D, y, max_nonzero):
    """
    Orthogonal Matching Pursuit (OMP) for sparse coding.

    Given a dictionary D and a signal y, find the sparsest
    coefficient vector x such that:
        y ≈ D · x,    with ||x||₀ ≤ max_nonzero

    OMP is a greedy algorithm that iteratively selects the
    dictionary atom most correlated with the current residual.

    Algorithm:
        1. Initialize residual r = y
        2. For each iteration (up to max_nonzero):
           a. Find atom most correlated with residual
           b. Add atom index to support set
           c. Solve least-squares: x_support = argmin ||y - D_support · x||²
           d. Update residual: r = y - D_support · x_support
        3. Return sparse coefficient vector x

    Parameters
    ----------
    D : np.ndarray
        Dictionary matrix (d × K), K atoms of dimension d.
    y : np.ndarray
        Signal vector (d,) — one patch to represent.
    max_nonzero : int
        Maximum number of non-zero coefficients (sparsity level).

    Returns
    -------
    x : np.ndarray
        Sparse coefficient vector (K,).
    """
    d, K = D.shape
    residual = y.copy()
    support = []       # Indices of selected atoms
    x = np.zeros(K)    # Sparse coefficients

    for iteration in range(max_nonzero):
        # ── Step A: Find most correlated atom ──
        # Compute inner product of each atom with the residual
        correlations = np.abs(D.T @ residual)

        # Mask already-selected atoms so we don't pick them again
        for idx in support:
            correlations[idx] = -1.0

        best_atom = np.argmax(correlations)

        # Stop if correlation is negligible (signal fully represented)
        if correlations[best_atom] < 1e-10:
            break

        support.append(best_atom)

        # ── Step B: Solve least-squares over support set ──
        # x_support = argmin ||y - D_support · x_support||²
        D_support = D[:, support]
        x_support, _, _, _ = np.linalg.lstsq(D_support, y, rcond=None)

        # ── Step C: Update residual ──
        residual = y - D_support @ x_support

        # Early stop if residual is very small
        if np.linalg.norm(residual) < 1e-6:
            break

    # Write the solution into the full coefficient vector
    if len(support) > 0:
        x[support] = x_support

    return x


# ═════════════════════════════════════════════════════════════
# STEP 4: Dictionary Update — The K-SVD Step
# ═════════════════════════════════════════════════════════════

def ksvd_dictionary_update(Y, D, X):
    """
    K-SVD dictionary update step.

    For each atom k in the dictionary:
        1. Find all signals that USE atom k (support set ω_k)
        2. Compute the error matrix WITHOUT atom k's contribution
        3. Use SVD to find the best rank-1 approximation
        4. Update atom k and its coefficients

    This is the core of K-SVD — it updates both the dictionary
    AND the sparse codes simultaneously using SVD.

    Parameters
    ----------
    Y : np.ndarray
        Data matrix (d × N) — centered patch vectors.
    D : np.ndarray
        Current dictionary (d × K).
    X : np.ndarray
        Current sparse codes (K × N).

    Returns
    -------
    D : np.ndarray
        Updated dictionary (d × K).
    X : np.ndarray
        Updated sparse codes (K × N).
    """
    D = D.copy()
    X = X.copy()
    K = D.shape[1]

    for k in range(K):
        # ── Find signals using atom k ──
        # ω_k = {i : X[k, i] ≠ 0}
        omega_k = np.where(np.abs(X[k, :]) > 1e-10)[0]

        if len(omega_k) == 0:
            # Atom k is unused — replace with a random direction
            # This prevents "dead" atoms in the dictionary
            D[:, k] = np.random.randn(D.shape[0])
            D[:, k] /= np.linalg.norm(D[:, k])
            continue

        # ── Compute error matrix excluding atom k ──
        # E_k = Y - Σ_{j≠k} d_j · x_j^T
        # Efficient form: E_k = Y - D·X + d_k · x_k^T
        E_k = (
            Y[:, omega_k]
            - D @ X[:, omega_k]
            + np.outer(D[:, k], X[k, omega_k])
        )

        # ── SVD of the restricted error matrix ──
        # E_k ≈ U · Σ · V^T
        # Best rank-1 approximation: σ₁ · u₁ · v₁^T
        U, S, Vt = np.linalg.svd(E_k, full_matrices=False)

        # ── Update atom k and its coefficients ──
        # New atom = first left singular vector (best direction)
        D[:, k] = U[:, 0]
        # New coefficients = first singular value × first right singular vector
        X[k, omega_k] = S[0] * Vt[0, :]

    return D, X


# ═════════════════════════════════════════════════════════════
# STEP 5: Complete K-SVD Denoising Pipeline
# ═════════════════════════════════════════════════════════════

def ksvd_denoise(image, alpha, patch_size=8, n_atoms=None,
                 iterations=10, progress_callback=None):
    """
    Complete K-SVD denoising pipeline.

    Pipeline:
        1. Extract overlapping patches from noisy image
        2. Center patches (subtract mean — preserves DC component)
        3. Initialize DCT dictionary
        4. Compute sparsity from alpha
        5. Run K-SVD iterations:
           a. Sparse coding (OMP on each patch)
           b. Dictionary update (SVD-based)
        6. Reconstruct patches: y_clean = D·x + mean
        7. Reconstruct image by averaging overlapping patches

    Alpha → Sparsity Mapping:
        max_nonzero = max(1, round((1 - α) × patch_size))

        α = 0.10 → (1-0.10) × 8 = 7 atoms → mild denoising
        α = 0.50 → (1-0.50) × 8 = 4 atoms → moderate denoising
        α = 0.90 → (1-0.90) × 8 = 1 atom  → strong denoising

    Parameters
    ----------
    image : np.ndarray
        Noisy grayscale image (uint8, values 0–255).
    alpha : float
        Sparsity control in [0.1, 1.0].
        Higher → fewer atoms → stronger denoising.
    patch_size : int
        Size of square patches (default: 8).
    n_atoms : int or None
        Number of dictionary atoms. Default = patch_size².
    iterations : int
        Number of K-SVD iterations (default: 10).
    progress_callback : callable or None
        Function(current_iter, total_iters) for progress updates.

    Returns
    -------
    dict
        Dictionary containing:
        - 'denoised'     : np.ndarray — Restored image (uint8)
        - 'dictionary'   : np.ndarray — Learned dictionary
        - 'sparsity'     : int        — Max non-zero coefficients
        - 'n_atoms'      : int        — Number of dictionary atoms
        - 'n_patches'    : int        — Number of patches processed
        - 'iterations'   : int        — K-SVD iterations performed
        - 'alpha'        : float      — Alpha value used
    """
    if n_atoms is None:
        n_atoms = patch_size * patch_size

    step = max(1, patch_size // 2)

    # ── Step 1: Extract overlapping patches ──
    img_float = image.astype(np.float64)
    Y, positions = extract_patches(img_float, patch_size, step)
    N = Y.shape[1]  # Number of patches

    # ── Step 2: Center patches (subtract per-patch mean) ──
    # This separates the DC component (brightness) from the
    # structure, helping the dictionary focus on textures/edges
    patch_means = Y.mean(axis=0, keepdims=True)
    Y_centered = Y - patch_means

    # ── Step 3: Initialize dictionary with DCT basis ──
    D = create_dct_dictionary(patch_size, n_atoms)

    # ── Step 4: Compute sparsity from alpha ──
    # Higher alpha → fewer atoms allowed → more denoising
    max_nonzero = max(1, int(round((1.0 - alpha) * patch_size)))

    # ── Step 5: K-SVD iterations ──
    for it in range(iterations):

        # ── 5a: Sparse Coding — OMP for each patch ──
        X = np.zeros((n_atoms, N))
        for i in range(N):
            X[:, i] = omp(D, Y_centered[:, i], max_nonzero)

        # ── 5b: Dictionary Update — K-SVD step ──
        D, X = ksvd_dictionary_update(Y_centered, D, X)

        # Report progress
        if progress_callback is not None:
            progress_callback(it + 1, iterations)

    # ── Step 6: Reconstruct patches ──
    # Clean patch = Dictionary × Sparse code + Mean
    Y_reconstructed = D @ X + patch_means

    # ── Step 7: Reconstruct image from patches ──
    denoised = reconstruct_from_patches(
        Y_reconstructed, positions, image.shape, patch_size
    )

    return {
        "denoised": denoised,
        "dictionary": D,
        "sparsity": max_nonzero,
        "n_atoms": n_atoms,
        "n_patches": N,
        "iterations": iterations,
        "alpha": alpha,
    }
