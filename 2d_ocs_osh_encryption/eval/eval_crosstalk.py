import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_PIXELS, WAVELENGTH, PIXEL_SIZE, TOLERANCE, DISTANCES, ITERATIONS, M_MEASUREMENTS
from modules.matrix_gen import generate_spe_matrices
from modules.encryption import OSH_encryption_eq21
from modules.decryption import trapdoor_extraction_eq33, solve_2dcs_twist
from main import calculate_cc, HAS_SKIMAGE
if HAS_SKIMAGE:
    from skimage import data
    from skimage.transform import resize

def run_crosstalk_analysis():
    os.makedirs('results/eval', exist_ok=True)
    L_images = 4
    plaintexts = []
    
    if HAS_SKIMAGE:
        images = [data.camera(), data.astronaut(), data.moon(), data.coins()]
        for img in images:
            img_r = resize(img, (N_PIXELS, N_PIXELS), anti_aliasing=True)
            if img_r.ndim == 3:
                img_r = np.mean(img_r, axis=2)
            img_r = img_r / np.max(img_r)
            plaintexts.append(img_r)
        plaintexts = plaintexts[:L_images]
    else:
        x = np.linspace(-1, 1, N_PIXELS)
        X, Y = np.meshgrid(x, x)
        plaintexts = [
            (np.sin(10 * X) + np.cos(10 * Y) + 2) / 4.0,
            np.exp(-(X**2 + Y**2) * 5),
            (np.sin(20 * X) + 1) / 2.0,
            (np.cos(20 * Y) + 1) / 2.0
        ]
        
    keys, Qa = generate_spe_matrices(N_PIXELS, M_MEASUREMENTS, L_images)
    ciphertext_I = OSH_encryption_eq21(plaintexts, keys, DISTANCES, WAVELENGTH, PIXEL_SIZE)
    
    cc_matrix = np.zeros((L_images, L_images))
    
    # We will attempt to decrypt each target channel l using each key k
    for l in range(L_images):
        for k in range(L_images):
            print(f"Decrypting channel {l} using key for channel {k}...")
            # Extract sub-ciphertext using key k
            Gamma_a, Phi_a, Phi_b = keys[k]['decryption_keys']
            I_h = trapdoor_extraction_eq33(ciphertext_I, Gamma_a)
            
            # Decrypt at distance z_l (corresponding to channel l)
            recovered_img = solve_2dcs_twist(
                I_h, Phi_a, Phi_b, N_PIXELS, WAVELENGTH, DISTANCES[l], PIXEL_SIZE, 
                iterations=200, tolerance=TOLERANCE  # 200 iterations for faster evaluation
            )
            
            # Measure similarity with original plaintext l
            cc = np.corrcoef(plaintexts[l].flatten(), recovered_img.flatten())[0, 1]
            cc_matrix[l, k] = cc
            print(f"  Channel {l} decrypted with Key {k} -> CC = {cc:.4f}")
            
    print("\nCrosstalk Correlation Matrix:")
    print(cc_matrix)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(cc_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Correlation Coefficient (CC)')
    plt.xticks(range(L_images), [f"Key {i}" for i in range(L_images)])
    plt.yticks(range(L_images), [f"Plaintext {i}" for i in range(L_images)])
    plt.title("Crosstalk Analysis Matrix")
    
    for i in range(L_images):
        for j in range(L_images):
            plt.text(j, i, f"{cc_matrix[i, j]:.4f}", ha='center', va='center', color='white' if cc_matrix[i, j] < 0.5 else 'black')
            
    plt.tight_layout()
    plt.savefig('results/eval/fig9_crosstalk_matrix.png')
    print("Saved Figure 9 to results/eval/fig9_crosstalk_matrix.png")

if __name__ == "__main__":
    run_crosstalk_analysis()
