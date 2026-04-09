"""
app.py — Comparative Image Restoration using K-SVD and DWT with Alpha Control
===============================================================================
Interactive teaching tool for comparing two image restoration approaches.

Features:
    - Image upload with auto-resize for performance
    - Gaussian noise addition (adjustable σ)
    - DWT denoising with wavelet subband visualization
    - K-SVD denoising with sparse dictionary learning
    - Alpha (α) internal control for both methods
    - Comparative analysis: metrics table, difference maps, PSNR vs α graph
    - Download restored images

Reference: Gonzalez & Woods, "Digital Image Processing"
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from utils import load_image_as_grayscale, add_gaussian_noise, image_to_bytes, resize_if_large, visualize_dictionary
from metrics import compute_all_metrics, compute_psnr, compute_difference_map
from wavelet import dwt_denoise, normalize_subband_for_display
from ksvd import ksvd_denoise


# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Restoration: K-SVD vs DWT",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom Styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .main-header p {
        color: #888;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }

    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-card .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .metric-value {
        color: #e2e8f0;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    /* Placeholder card for future methods */
    .placeholder-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 2px dashed #475569;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        color: #64748b;
    }
    .placeholder-card h3 {
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    .placeholder-card p {
        font-size: 0.9rem;
    }

    /* Image container */
    .image-container {
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.5rem;
        background: #0f172a;
    }

    /* Sidebar section headers */
    .sidebar-section {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #334155;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ready {
        background: #064e3b;
        color: #6ee7b7;
    }
    .status-pending {
        background: #78350f;
        color: #fcd34d;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Comparative Image Restoration</h1>
    <p>K-SVD (Sparse Representation) vs DWT (Wavelet Thresholding) with Internal Alpha Control</p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────────────────────
# Sidebar — Controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    # ── Image Upload ──
    st.markdown('<div class="sidebar-section"> Image Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload a grayscale or color image. It will be converted to grayscale automatically."
    )

    # ── Noise Parameters ──
    st.markdown('<div class="sidebar-section"> Noise Parameters</div>', unsafe_allow_html=True)
    sigma = st.slider(
        "Noise Level (σ)",
        min_value=0,
        max_value=50,
        value=25,
        step=1,
        help="Standard deviation of Gaussian noise. σ=0 means no noise, σ=50 is heavy noise."
    )

    # ── Restoration Parameters ──
    st.markdown('<div class="sidebar-section"> Restoration Parameters</div>', unsafe_allow_html=True)
    alpha = st.slider(
        "Alpha (α)",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Controls denoising strength INSIDE each method. DWT → threshold = α × max|coeff|. K-SVD → scaling atoms."
    )

    patch_size = st.selectbox(
        "Patch Size",
        options=[4, 8, 16],
        index=1,
        help="Size of image patches for K-SVD dictionary learning (default: 8×8)."
    )

    iterations = st.selectbox(
        "K-SVD Iterations",
        options=[5, 10, 15, 20, 30],
        index=1,
        help="Number of dictionary learning iterations for K-SVD (default: 10)."
    )

    # Alpha live preview in sidebar
    max_nonzero_preview = max(1, int(round((1.0 - alpha) * (patch_size ** 2 / 4))))
    st.markdown(
        f'<div style="background:#1e293b; border-left:3px solid #667eea; '
        f'padding:0.5rem 0.75rem; border-radius:4px; margin-top:0.25rem; font-size:0.78rem; color:#94a3b8;">'
        f'<b>α = {alpha:.2f}</b><br>'
        f'DWT: T = {alpha:.2f} × max|coeff|<br>'
        f'K-SVD: {max_nonzero_preview} atoms/patch'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Phase Status ──
    st.markdown('<div class="sidebar-section"> Phase Status</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="status-badge status-ready">✓ Phase 1 — UI & Noise</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span class="status-badge status-ready">✓ Phase 2 — DWT</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span class="status-badge status-ready">✓ Phase 3 — K-SVD</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span class="status-badge status-ready">✓ Phase 4 — Alpha Control</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span class="status-badge status-ready">✓ Phase 5 — Comparison</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span class="status-badge status-ready">✓ Phase 6 — Final</span>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# Main Panel
# ─────────────────────────────────────────────────────────────
if uploaded_file is not None:
    # Load, resize if needed, and process image
    original_image = load_image_as_grayscale(uploaded_file)
    orig_h, orig_w = original_image.shape
    original_image = resize_if_large(original_image, max_dim=300)
    if original_image.shape != (orig_h, orig_w):
        st.info(
            f"📐 Image resized from {orig_w}×{orig_h} to "
            f"{original_image.shape[1]}×{original_image.shape[0]} "
            f"for faster K-SVD processing."
        )
    noisy_image = add_gaussian_noise(original_image, sigma=sigma)

    # Compute metrics for noisy image vs original
    noise_metrics = compute_all_metrics(original_image, noisy_image)

    # ── Row 1: Original + Noisy ──
    st.subheader(" Input Images")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image (Grayscale)**")
        st.image(
            original_image,
            use_container_width=True,
            clamp=True,
        )
        st.caption(f"Size: {original_image.shape[1]}×{original_image.shape[0]} px")

    with col2:
        st.markdown(f"**Noisy Image (σ = {sigma})**")
        st.image(
            noisy_image,
            use_container_width=True,
            clamp=True,
        )
        st.caption(f"Noise model: g(x,y) = f(x,y) + η,  η ~ N(0, {sigma}²)")

    # ── Noise Metrics ──
    st.subheader("Noise Impact Metrics")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PSNR</div>
            <div class="metric-value">{noise_metrics['PSNR']:.2f} dB</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MSE</div>
            <div class="metric-value">{noise_metrics['MSE']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SSIM</div>
            <div class="metric-value">{noise_metrics['SSIM']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Row 2: Restoration Outputs ──
    st.subheader(" Restoration Outputs")

    # Run DWT first so we can use threshold in the banner
    dwt_result = dwt_denoise(noisy_image, alpha=alpha, wavelet="db1")
    dwt_denoised = dwt_result["denoised"]
    max_nonzero_display = max(1, int(round((1.0 - alpha) * (patch_size ** 2 / 4))))

    # Alpha explanation banner
    st.markdown(
        f'<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); '
        f'border: 1px solid #334155; border-radius: 12px; padding: 1rem 1.5rem; '
        f'margin-bottom: 1rem; text-align: center;">'
        f'<span style="color: #a78bfa; font-weight: 700; font-size: 0.95rem;">'
        f'🎛️ Alpha (α = {alpha:.2f}) controls denoising strength inside each method independently</span>'
        f'<div style="display: flex; justify-content: center; gap: 3rem; margin-top: 0.75rem;">'
        f'<div style="color: #94a3b8; font-size: 0.82rem;">'
        f'<b style="color: #60a5fa;">DWT</b> → Threshold T = {alpha:.2f} × max|coefficients| = <b>{dwt_result["threshold"]:.1f}</b>'
        f'</div>'
        f'<div style="color: #94a3b8; font-size: 0.82rem;">'
        f'<b style="color: #f97316;">K-SVD</b> → max {max_nonzero_display} non-zero atoms per patch (of {patch_size}²={patch_size*patch_size})'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    col3, col4 = st.columns(2)

    # ── DWT Denoising (LIVE) ──
    with col3:
        st.markdown(f"** DWT Denoised (α = {alpha:.2f})**")

        st.image(
            dwt_denoised,
            use_container_width=True,
            clamp=True,
        )
        st.caption(
            f"Wavelet: {dwt_result['wavelet']} | "
            f"T = α·max|c| = {alpha:.2f}×{dwt_result['threshold']/alpha:.0f} = {dwt_result['threshold']:.2f} | "
            f"Higher α → stronger thresholding"
        )

        # Download button for DWT
        st.download_button(
            "⬇️ Download DWT Result",
            data=image_to_bytes(dwt_denoised),
            file_name=f"dwt_denoised_alpha{alpha:.2f}.png",
            mime="image/png",
            use_container_width=True,
        )

        # DWT quality metrics
        dwt_metrics = compute_all_metrics(original_image, dwt_denoised)
        dm1, dm2, dm3 = st.columns(3)
        with dm1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PSNR</div>
                <div class="metric-value">{dwt_metrics['PSNR']:.2f} dB</div>
            </div>
            """, unsafe_allow_html=True)
        with dm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MSE</div>
                <div class="metric-value">{dwt_metrics['MSE']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with dm3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">SSIM</div>
                <div class="metric-value">{dwt_metrics['SSIM']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── K-SVD Denoising (LIVE) ──
    with col4:
        st.markdown(f"** K-SVD Denoised (α = {alpha:.2f})**")

        # Compute sparsity for display
        max_nonzero = max(1, int(round((1.0 - alpha) * patch_size)))

        # Run K-SVD with caching and progress indicator
        @st.cache_data(show_spinner=False)
        def run_ksvd_cached(noisy, alpha_val, p_size, n_atoms_val, iters):
            return ksvd_denoise(
                noisy, alpha=alpha_val, patch_size=p_size,
                n_atoms=n_atoms_val, iterations=iters
            )

        n_atoms = patch_size * patch_size
        with st.spinner(f"Running K-SVD ({iterations} iterations, {max_nonzero} atoms/patch)..."):
            ksvd_result = run_ksvd_cached(
                noisy_image, alpha, patch_size, n_atoms, iterations
            )
        ksvd_denoised = ksvd_result["denoised"]

        st.image(
            ksvd_denoised,
            use_container_width=True,
            clamp=True,
        )
        st.caption(
            f"Atoms: {ksvd_result['n_atoms']} | "
            f"Active: {ksvd_result['sparsity']} = max(1, round((1−{alpha:.2f})×{patch_size}²/4)) | "
            f"Higher α → fewer atoms → sparser"
        )

        # Download button for K-SVD
        st.download_button(
            "⬇️ Download K-SVD Result",
            data=image_to_bytes(ksvd_denoised),
            file_name=f"ksvd_denoised_alpha{alpha:.2f}.png",
            mime="image/png",
            use_container_width=True,
        )

        # K-SVD quality metrics
        ksvd_metrics = compute_all_metrics(original_image, ksvd_denoised)
        km1, km2, km3 = st.columns(3)
        with km1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PSNR</div>
                <div class="metric-value">{ksvd_metrics['PSNR']:.2f} dB</div>
            </div>
            """, unsafe_allow_html=True)
        with km2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MSE</div>
                <div class="metric-value">{ksvd_metrics['MSE']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with km3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">SSIM</div>
                <div class="metric-value">{ksvd_metrics['SSIM']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════
    # COMPARATIVE ANALYSIS SECTION (Phase 5)
    # ═════════════════════════════════════════════════════════════
    st.divider()
    st.subheader(" Comparative Analysis")

    # ── Side-by-side: All 4 images ──
    st.markdown("##### Visual Comparison")
    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1:
        st.image(original_image, caption="Original", use_container_width=True, clamp=True)
    with vc2:
        st.image(noisy_image, caption=f"Noisy (σ={sigma})", use_container_width=True, clamp=True)
    with vc3:
        st.image(dwt_denoised, caption=f"DWT (α={alpha:.2f})", use_container_width=True, clamp=True)
    with vc4:
        st.image(ksvd_denoised, caption=f"K-SVD (α={alpha:.2f})", use_container_width=True, clamp=True)

    # ── Metrics Comparison Table ──
    st.markdown("##### Numerical Comparison")

    # Determine winners for each metric
    psnr_winner = "DWT" if dwt_metrics['PSNR'] >= ksvd_metrics['PSNR'] else "K-SVD"
    mse_winner = "DWT" if dwt_metrics['MSE'] <= ksvd_metrics['MSE'] else "K-SVD"
    ssim_winner = "DWT" if dwt_metrics['SSIM'] >= ksvd_metrics['SSIM'] else "K-SVD"

    def highlight_val(val, is_winner):
        """Wrap value with green highlight if winner."""
        if is_winner:
            return f'<b style="color: #6ee7b7;">{val}</b>'
        return f'{val}'

    st.markdown(f"""
    <table style="width:100%; border-collapse:collapse; text-align:center; font-size:0.9rem;">
        <tr style="background:#1e293b; color:#94a3b8;">
            <th style="padding:0.6rem; border:1px solid #334155;">Metric</th>
            <th style="padding:0.6rem; border:1px solid #334155;">Noisy</th>
            <th style="padding:0.6rem; border:1px solid #334155;">🌊 DWT</th>
            <th style="padding:0.6rem; border:1px solid #334155;">📖 K-SVD</th>
            <th style="padding:0.6rem; border:1px solid #334155;">🏆 Winner</th>
        </tr>
        <tr style="background:#0f172a; color:#e2e8f0;">
            <td style="padding:0.5rem; border:1px solid #334155; font-weight:600;">PSNR (dB) ↑</td>
            <td style="padding:0.5rem; border:1px solid #334155;">{noise_metrics['PSNR']:.2f}</td>
            <td style="padding:0.5rem; border:1px solid #334155;">{highlight_val(f"{dwt_metrics['PSNR']:.2f}", psnr_winner=='DWT')}</td>
            <td style="padding:0.5rem; border:1px solid #334155;">{highlight_val(f"{ksvd_metrics['PSNR']:.2f}", psnr_winner=='K-SVD')}</td>
            <td style="padding:0.5rem; border:1px solid #334155; color:#fbbf24;">{psnr_winner}</td>
        </tr>
        <tr style="background:#1a1a2e; color:#e2e8f0;">
            <td style="padding:0.5rem; border:1px solid #334155; font-weight:600;">MSE ↓</td>
            <td style="padding:0.5rem; border:1px solid #334155;">{noise_metrics['MSE']:.2f}</td>
            <td style="padding:0.5rem; border:1px solid #334155;">{highlight_val(f"{dwt_metrics['MSE']:.2f}", mse_winner=='DWT')}</td>
            <td style="padding:0.5rem; border:1px solid #334155;">{highlight_val(f"{ksvd_metrics['MSE']:.2f}", mse_winner=='K-SVD')}</td>
            <td style="padding:0.5rem; border:1px solid #334155; color:#fbbf24;">{mse_winner}</td>
        </tr>
        <tr style="background:#0f172a; color:#e2e8f0;">
            <td style="padding:0.5rem; border:1px solid #334155; font-weight:600;">SSIM ↑</td>
            <td style="padding:0.5rem; border:1px solid #334155;">{noise_metrics['SSIM']:.4f}</td>
            <td style="padding:0.5rem; border:1px solid #334155;">{highlight_val(f"{dwt_metrics['SSIM']:.4f}", ssim_winner=='DWT')}</td>
            <td style="padding:0.5rem; border:1px solid #334155;">{highlight_val(f"{ksvd_metrics['SSIM']:.4f}", ssim_winner=='K-SVD')}</td>
            <td style="padding:0.5rem; border:1px solid #334155; color:#fbbf24;">{ssim_winner}</td>
        </tr>
    </table>
    <p style="text-align:center; color:#64748b; font-size:0.75rem; margin-top:0.5rem;">
        ↑ = higher is better &nbsp;&nbsp;|&nbsp;&nbsp; ↓ = lower is better &nbsp;&nbsp;|&nbsp;&nbsp;
        <span style="color:#6ee7b7;">green</span> = winner for that metric
    </p>
    """, unsafe_allow_html=True)

    # ── Difference Maps ──
    st.markdown("##### Difference Maps  |Original − Restored|")
    st.caption("Bright regions indicate large restoration error. Differences are amplified 3× for visibility.")

    diff_noisy = compute_difference_map(original_image, noisy_image, amplify=3.0)
    diff_dwt = compute_difference_map(original_image, dwt_denoised, amplify=3.0)
    diff_ksvd = compute_difference_map(original_image, ksvd_denoised, amplify=3.0)

    # Use matplotlib for heatmap-style difference maps
    fig_diff, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig_diff.patch.set_facecolor('#0e1117')

    for ax, diff_img, title in zip(
        axes,
        [diff_noisy, diff_dwt, diff_ksvd],
        [f"Noisy (σ={sigma})", f"DWT (α={alpha:.2f})", f"K-SVD (α={alpha:.2f})"]
    ):
        im = ax.imshow(diff_img, cmap='inferno', vmin=0, vmax=150)
        ax.set_title(title, color='#94a3b8', fontsize=10, fontweight='bold')
        ax.axis('off')

    fig_diff.colorbar(im, ax=axes, shrink=0.8, label='Error (amplified)',
                      orientation='vertical', pad=0.02)
    fig_diff.tight_layout(pad=1.0)
    st.pyplot(fig_diff)
    plt.close(fig_diff)

    # ── PSNR vs Alpha Graph ──
    st.markdown("##### PSNR vs Alpha (α)")
    st.caption("DWT: full sweep (fast) | K-SVD: current α point (full sweep is computationally expensive)")

    @st.cache_data(show_spinner=False)
    def compute_dwt_alpha_sweep(noisy_img, orig_img):
        """Compute DWT PSNR across all alpha values (fast)."""
        alpha_values = np.arange(0.05, 1.01, 0.05)
        psnr_values = []
        for a in alpha_values:
            result = dwt_denoise(noisy_img, alpha=a)
            psnr_val = compute_psnr(orig_img, result['denoised'])
            psnr_values.append(psnr_val)
        return alpha_values, psnr_values

    dwt_alphas, dwt_psnrs = compute_dwt_alpha_sweep(noisy_image, original_image)

    # Create PSNR vs Alpha plot
    fig_psnr, ax_psnr = plt.subplots(figsize=(10, 4.5))
    fig_psnr.patch.set_facecolor('#0e1117')
    ax_psnr.set_facecolor('#1a1a2e')

    # DWT curve
    ax_psnr.plot(dwt_alphas, dwt_psnrs, 'o-',
                 color='#60a5fa', linewidth=2, markersize=5,
                 label='DWT (Wavelet)', zorder=3)

    # K-SVD current point
    ax_psnr.plot(alpha, ksvd_metrics['PSNR'], 's',
                 color='#f97316', markersize=12, markeredgecolor='white',
                 markeredgewidth=2, label=f'K-SVD (α={alpha:.2f})', zorder=4)

    # DWT current point highlight
    ax_psnr.plot(alpha, dwt_metrics['PSNR'], 'D',
                 color='#60a5fa', markersize=12, markeredgecolor='white',
                 markeredgewidth=2, label=f'DWT (α={alpha:.2f})', zorder=4)

    # Noisy baseline
    ax_psnr.axhline(y=noise_metrics['PSNR'], color='#ef4444',
                    linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'Noisy baseline ({noise_metrics["PSNR"]:.1f} dB)')

    # Current alpha vertical line
    ax_psnr.axvline(x=alpha, color='#a78bfa', linestyle=':',
                    linewidth=1, alpha=0.5)

    # Styling
    ax_psnr.set_xlabel('Alpha (α)', color='#94a3b8', fontsize=11)
    ax_psnr.set_ylabel('PSNR (dB)', color='#94a3b8', fontsize=11)
    ax_psnr.set_title('Denoising Quality vs Alpha', color='#e2e8f0',
                      fontsize=13, fontweight='bold')
    ax_psnr.legend(loc='best', fontsize=9, facecolor='#1e293b',
                   edgecolor='#334155', labelcolor='#e2e8f0')
    ax_psnr.tick_params(colors='#94a3b8')
    ax_psnr.grid(True, alpha=0.2, color='#475569')
    ax_psnr.set_xlim(0, 1.05)

    for spine in ax_psnr.spines.values():
        spine.set_color('#334155')

    fig_psnr.tight_layout()
    st.pyplot(fig_psnr)
    plt.close(fig_psnr)

    st.info(
        " **Reading the graph:** The DWT curve shows how PSNR changes across all alpha values. "
        "The K-SVD point shows quality at the current α. Both methods should peak at some optimal α — "
        "too low preserves noise, too high removes detail."
    )

    st.divider()

    # ── Wavelet Subband Visualization ──
    with st.expander(" DWT Subband Visualization (Educational)", expanded=False):
        st.markdown("""
        The DWT decomposes the image into **four subbands** (Gonzalez, Ch. 7):
        - **LL** — Approximation (low-frequency structure)
        - **LH** — Horizontal details (horizontal edges)
        - **HL** — Vertical details (vertical edges)
        - **HH** — Diagonal details (noise + diagonal edges)

        Soft thresholding is applied to **LH, HL, HH only** — the LL subband
        is preserved to maintain image structure.
        """)

        # Before thresholding
        st.markdown("##### Before Thresholding (Noisy Subbands)")
        sb1, sb2, sb3, sb4 = st.columns(4)
        with sb1:
            st.image(
                normalize_subband_for_display(dwt_result["LL"]),
                caption="LL (Approx)",
                use_container_width=True,
            )
        with sb2:
            st.image(
                normalize_subband_for_display(dwt_result["LH"]),
                caption="LH (Horiz)",
                use_container_width=True,
            )
        with sb3:
            st.image(
                normalize_subband_for_display(dwt_result["HL"]),
                caption="HL (Vert)",
                use_container_width=True,
            )
        with sb4:
            st.image(
                normalize_subband_for_display(dwt_result["HH"]),
                caption="HH (Diag)",
                use_container_width=True,
            )

        # After thresholding
        st.markdown(f"##### After Soft Thresholding (T = {dwt_result['threshold']:.2f})")
        sa1, sa2, sa3, sa4 = st.columns(4)
        with sa1:
            st.image(
                normalize_subband_for_display(dwt_result["LL"]),
                caption="LL (Unchanged)",
                use_container_width=True,
            )
        with sa2:
            st.image(
                normalize_subband_for_display(dwt_result["LH_thresh"]),
                caption="LH (Thresholded)",
                use_container_width=True,
            )
        with sa3:
            st.image(
                normalize_subband_for_display(dwt_result["HL_thresh"]),
                caption="HL (Thresholded)",
                use_container_width=True,
            )
        with sa4:
            st.image(
                normalize_subband_for_display(dwt_result["HH_thresh"]),
                caption="HH (Thresholded)",
                use_container_width=True,
            )

        # Explain what happened
        st.info(
            f"**Threshold T = {dwt_result['threshold']:.2f}** "
            f"(α={alpha:.2f} × max|coefficients|). "
            f"Coefficients with |w| ≤ T are zeroed out. "
            f"Notice how the detail subbands become sparser after thresholding — "
            f"this removes noise while preserving strong edges."
        )

    # ── Learned Dictionary Visualization ──
    with st.expander("📖 Learned Dictionary (K-SVD Atoms)", expanded=False):
        st.markdown(f"""
        This grid shows the **{ksvd_result['n_atoms']} dictionary atoms** learned by K-SVD to sparsely represent patches
        from this specific image. Notice how some atoms capture flat regions, while others capture edges and textures.
        """)
        
        show_dictionary = st.checkbox("Show Dictionary Atoms Grid", value=False)
        if show_dictionary:
            dict_grid = visualize_dictionary(ksvd_result["dictionary"], patch_size)
            
            st.image(
                dict_grid, 
                caption=f"K-SVD Dictionary ({patch_size}x{patch_size} atoms)", 
                use_container_width=True, 
                clamp=True
            )
            
            st.download_button(
                "⬇️ Download Dictionary Grid",
                data=image_to_bytes(dict_grid),
                file_name=f"ksvd_dictionary_alpha{alpha:.2f}.png",
                mime="image/png",
                use_container_width=True,
            )

    # ── Current Parameters Summary ──
    st.divider()
    st.subheader(" Current Parameters")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Noise σ", sigma)
    p2.metric("Alpha α", f"{alpha:.2f}")
    p3.metric("Patch Size", f"{patch_size}×{patch_size}")
    p4.metric("Iterations", iterations)

    # Alpha interpretation panel
    with st.expander(" How Alpha (α) Works in Each Method", expanded=False):
        acol1, acol2 = st.columns(2)
        with acol1:
            st.markdown(f"""
            #### 🌊 DWT (Wavelet Thresholding)
            **Formula:** `T = α × max(|detail coefficients|)`

            - α directly scales the threshold T
            - **Low α (0.1)** → small T → few coefficients zeroed → **mild denoising**
            - **High α (1.0)** → large T → many coefficients zeroed → **strong denoising**

            Current: **T = {alpha:.2f} × {dwt_result['threshold']/alpha:.0f} = {dwt_result['threshold']:.1f}**
            """)
        with acol2:
            st.markdown(f"""
            ####  K-SVD (Sparse Representation)
            **Formula:** `max_atoms = max(1, round((1 − α) × patch_size² / 4))`

            - α inversely controls how many dictionary atoms each patch can use
            - **Low α (0.1)** → many atoms → detailed reconstruction → **mild denoising**
            - **High α (1.0)** → few atoms → coarse reconstruction → **strong denoising**

            Current: **max_atoms = max(1, round((1−{alpha:.2f}) × {patch_size}² / 4)) = {ksvd_result['sparsity']}**
            """)
        st.warning(
            " **Important:** Alpha acts INSIDE each method independently. "
            "It is NOT used to compare methods directly — it controls denoising strength within each algorithm."
        )

else:
    # ── Welcome screen when no image is uploaded ──
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem 1rem 2rem;">
        <h2 style="color: #94a3b8;">👆 Upload an image to get started</h2>
        <p style="color: #64748b; max-width: 600px; margin: 1rem auto;">
            An interactive teaching tool for comparing two fundamentally different
            image restoration techniques with internal alpha control.
        </p>
        <p style="color: #64748b; font-size: 0.9rem;">
            Supported formats: PNG, JPG, JPEG, BMP, TIFF
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Theory: What is DWT? ──
    with st.expander(" What is DWT (Discrete Wavelet Transform)?", expanded=True):
        st.markdown("""
        **DWT** operates in the **frequency domain** by decomposing an image into
        subbands at different scales and orientations (Gonzalez, Ch. 7).

        **How it denoises:**
        1. Apply 2D DWT → decompose into **LL** (approximation) + **LH, HL, HH** (details)
        2. Noise primarily resides in the high-frequency detail subbands
        3. Apply **soft thresholding** to LH, HL, HH:
           - Coefficients with |w| ≤ T are set to zero (noise removed)
           - Coefficients with |w| > T are shrunk toward zero
        4. Inverse DWT → reconstruct clean image

        **Threshold formula:** `T = α × max(|detail coefficients|)`

        **Key properties:**
        - ✅ Very fast (single pass)
        - ✅ Works well for smooth images
        - ❌ May over-smooth textures
        - ❌ Fixed basis (not adaptive to image content)
        """)

    # ── Theory: What is K-SVD? ──
    with st.expander("📖 What is K-SVD (Dictionary Learning)?", expanded=True):
        st.markdown("""
        **K-SVD** operates in the **spatial domain** by learning a dictionary of
        patterns (atoms) that sparsely represent image patches (Aharon et al., 2006).

        **How it denoises:**
        1. Extract overlapping patches from the noisy image
        2. Initialize dictionary with DCT basis functions
        3. For each patch, find a **sparse** combination of dictionary atoms
           using **OMP** (Orthogonal Matching Pursuit)
        4. Update dictionary atoms using **SVD** (the "K-SVD" step)
        5. Repeat steps 3-4 for multiple iterations
        6. Reconstruct image by averaging overlapping clean patches

        **Sparsity formula:** `max_atoms = max(1, round((1 − α) × patch_size))`

        **Key properties:**
        - ✅ Learns patterns specific to the image
        - ✅ Better at preserving textures and edges
        - ❌ Slower (iterative, many patches)
        - ❌ Results depend on initialization and parameters
        """)

    # ── Theory: Role of Alpha ──
    with st.expander("🎛️ Role of Alpha (α) — Internal Denoising Control", expanded=True):
        st.markdown("""
        **Alpha is NOT used to compare the two methods.** Instead, it acts
        **INSIDE** each method independently to control denoising strength.

        | α Value | DWT Effect | K-SVD Effect |
        |---------|------------|---------------|
        | **Low (0.1)** | Small threshold → mild denoising | Many atoms → detailed reconstruction |
        | **Mid (0.5)** | Moderate threshold | Moderate sparsity |
        | **High (1.0)** | Large threshold → aggressive denoising | Few atoms → coarse reconstruction |

        This allows students to see how each method **responds to the same parameter**
        and understand their internal behavior.
        """)

    # ── Noise Model and Metrics ──
    with st.expander(" Noise Model & Quality Metrics", expanded=False):
        st.markdown("""
        ### Gaussian Noise Model (Gonzalez, Ch. 5)

        > **g(x, y) = f(x, y) + η(x, y)** &nbsp;&nbsp; where η ~ N(0, σ²)

        | Symbol | Meaning |
        |--------|---------|
        | f(x,y) | Original (clean) image |
        | η(x,y) | Gaussian noise (zero mean, std = σ) |
        | g(x,y) | Observed noisy image |
        | σ | Noise standard deviation (0 = none, 50 = heavy) |

        ### Quality Metrics

        | Metric | Formula | Interpretation |
        |--------|---------|----------------|
        | **PSNR** | 10·log₁₀(255²/MSE) dB | Higher = better (30+ dB is good) |
        | **MSE** | mean((f − g)²) | Lower = better (0 = identical) |
        | **SSIM** | Structural similarity | Closer to 1 = better |
        """)

    # ── Comparison Table ──
    with st.expander("🔍 DWT vs K-SVD at a Glance", expanded=False):
        st.markdown("""
        | Feature | DWT (Wavelet) | K-SVD (Dictionary) |
        |---------|---------------|---------------------|
        | **Domain** | Frequency | Spatial |
        | **Basis** | Fixed (wavelet functions) | Learned (adapted to image) |
        | **Method** | Coefficient thresholding | Sparse coding + dictionary update |
        | **α controls** | Thresholding strength | Sparsity (active atoms) |
        | **Speed** | ⚡ Very fast | 🐢 Slower (iterative) |
        | **Textures** | May over-smooth | Better preserved |
        | **Edges** | Good | Very good |
        """)
