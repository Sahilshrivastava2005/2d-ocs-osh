import numpy as np
import matplotlib.pyplot as plt
import os

def eval_security():
    os.makedirs('results/eval', exist_ok=True)
    
    # Load ciphertext
    try:
        ciphertext = np.load('results/ciphertext.npy')
    except FileNotFoundError:
        print("Error: ciphertext.npy not found. Run main.py first.")
        return
        
    real_part = np.real(ciphertext).flatten()
    imag_part = np.imag(ciphertext).flatten()
    
    # 1. Figure 7: Gaussian Distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(real_part, bins=100, color='blue', alpha=0.7, density=True)
    plt.title("Distribution of Real Part")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(imag_part, bins=100, color='red', alpha=0.7, density=True)
    plt.title("Distribution of Imaginary Part")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig('results/eval/fig7_gaussian_dist.png')
    print("Saved Figure 7: Gaussian Distribution to results/eval/fig7_gaussian_dist.png")
    
    # 2. Figure 8: Adjacent Pixel Correlation
    # We will analyze magnitude, real, and imaginary parts separately
    mag = np.abs(ciphertext)
    real_val = np.real(ciphertext)
    imag_val = np.imag(ciphertext)
    
    # Calculate adjacent pixel correlations
    cc_mag = np.corrcoef(mag.flatten()[:-1], mag.flatten()[1:])[0, 1]
    cc_real = np.corrcoef(real_val.flatten()[:-1], real_val.flatten()[1:])[0, 1]
    cc_imag = np.corrcoef(imag_val.flatten()[:-1], imag_val.flatten()[1:])[0, 1]
    
    print(f"Adjacent Pixel Correlation Coefficient (Magnitude): {cc_mag:.4f}")
    print(f"Adjacent Pixel Correlation Coefficient (Real Part): {cc_real:.4f}")
    print(f"Adjacent Pixel Correlation Coefficient (Imaginary Part): {cc_imag:.4f}")
    
    # Scatter plots for real and imaginary parts
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(mag.flatten()[:5000], mag.flatten()[1:5001], s=1, alpha=0.5, color='green')
    plt.title(f"Magnitude (CC = {cc_mag:.4f})")
    plt.xlabel("I(x, y)")
    plt.ylabel("I(x+1, y)")
    
    plt.subplot(1, 3, 2)
    plt.scatter(real_val.flatten()[:5000], real_val.flatten()[1:5001], s=1, alpha=0.5, color='blue')
    plt.title(f"Real Part (CC = {cc_real:.4f})")
    plt.xlabel("Real(x, y)")
    plt.ylabel("Real(x+1, y)")
    
    plt.subplot(1, 3, 3)
    plt.scatter(imag_val.flatten()[:5000], imag_val.flatten()[1:5001], s=1, alpha=0.5, color='red')
    plt.title(f"Imaginary Part (CC = {cc_imag:.4f})")
    plt.xlabel("Imag(x, y)")
    plt.ylabel("Imag(x+1, y)")
    
    plt.tight_layout()
    plt.savefig('results/eval/fig8_pixel_correlation.png')
    print("Saved Figure 8: Pixel Correlation to results/eval/fig8_pixel_correlation.png")

if __name__ == "__main__":
    eval_security()
