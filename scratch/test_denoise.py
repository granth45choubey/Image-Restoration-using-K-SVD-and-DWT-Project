
import numpy as np
from utils import add_gaussian_noise
from wavelet import dwt_denoise
from ksvd import ksvd_denoise
from metrics import compute_all_metrics

def test():
    # Create a small dummy image
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    noisy = add_gaussian_noise(img, sigma=20)
    
    print("Testing DWT...")
    dwt_res = dwt_denoise(noisy, alpha=0.5)
    print("DWT PSNR:", compute_all_metrics(img, dwt_res['denoised'])['PSNR'])
    
    print("Testing K-SVD...")
    ksvd_res = ksvd_denoise(noisy, alpha=0.5, iterations=2, patch_size=8)
    print("K-SVD PSNR:", compute_all_metrics(img, ksvd_res['denoised'])['PSNR'])
    
    print("All tests passed!")

if __name__ == "__main__":
    test()
