import numpy as np
from scipy import ndimage
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim

def evaluate_denoising(I: np.ndarray, D: np.ndarray, eps: float = 1e-10) -> dict:
    """
    I : original/noisy image (2D numpy array)
    D : denoised image (2D numpy array)

    Returns dictionary with:
    - Noise Reduction Rate
    - ENL increase
    - Residual mean
    - Edge Preservation Index (EPI)
    - High Frequency Energy Ratio
    - Structure in residual (SSIM)
    - Histogram similarity
    """

    I = I.astype(np.float64)
    D = D.astype(np.float64)

    residual = I - D

    # --- 1. Noise Reduction Rate
    var_I = np.var(I)
    var_R = np.var(residual)
    nrr = 1 - (var_R / (var_I + eps))

    # --- 2. Equivalent Number of Looks (ENL)
    def enl(im):
        m = np.mean(im)
        v = np.var(im)
        return (m ** 2) / (v + eps)

    enl_I = enl(I)
    enl_D = enl(D)
    enl_increase = (enl_D - enl_I) / (enl_I + eps)

    # --- 3. Residual mean
    residual_mean = np.mean(residual)

    # --- 4. Edge Preservation Index (correlation of Sobel edges)
    G_I = sobel(I)
    G_D = sobel(D)

    epi_num = np.sum((G_I - G_I.mean()) * (G_D - G_D.mean()))
    epi_den = np.sqrt(np.sum((G_I - G_I.mean())**2) * np.sum((G_D - G_D.mean())**2)) + eps
    epi = epi_num / epi_den

    # --- 5. High Frequency Energy ratio
    def high_freq_energy(im):
        low = ndimage.gaussian_filter(im, sigma=2)
        high = im - low
        return np.sum(high ** 2)

    hf_I = high_freq_energy(I)
    hf_D = high_freq_energy(D)

    hf_ratio = hf_D / (hf_I + eps)

    # --- 6. Structural content in residual (SSIM)
    h, w = residual.shape
    ssim_residual = ssim(residual, np.zeros((h, w)), data_range=residual.max() - residual.min())

    # --- 7. Histogram similarity (correlation)
    hI, _ = np.histogram(I.flatten(), bins=256, range=(0, 255), density=True)
    hD, _ = np.histogram(D.flatten(), bins=256, range=(0, 255), density=True)

    hist_corr = np.corrcoef(hI, hD)[0, 1]

    return {
        "Noise reduction rate (ideal 0.4–0.85)": round(float(nrr), 4),
        "ENL original": round(float(enl_I), 3),
        "ENL denoised": round(float(enl_D), 3),
        "ENL relative increase (ideal > 1.0)": round(float(enl_increase), 4),
        "Residual mean (ideal ≈ 0)": round(float(residual_mean), 6),
        "Edge preservation index (ideal 0.8–1.05)": round(float(epi), 4),
        "HF energy ratio (ideal 0.3–0.7)": round(float(hf_ratio), 4),
        "Residual structure SSIM (ideal ≈ 0)": round(float(ssim_residual), 6),
        "Histogram similarity (ideal > 0.9)": round(float(hist_corr), 4),
    }
