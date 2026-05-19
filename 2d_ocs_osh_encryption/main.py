import numpy as np
import os
import matplotlib.pyplot as plt
try:
    from skimage.data import camera
    from skimage.transform import resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from config import N_PIXELS, M_MEASUREMENTS, WAVELENGTH, PIXEL_SIZE, DISTANCES, ITERATIONS, TOLERANCE
from modules.matrix_gen import generate_spe_matrices, generate_structural_patterns
from modules.encryption import OSH_encryption_eq21
from modules.decryption import trapdoor_extraction_eq33, solve_2dcs_twist

def calculate_cc(img1, img2):
    """ Calculates Correlation Coefficient between two images. """
    img1_mean = np.mean(img1)
    img2_mean = np.mean(img2)
    numerator = np.sum((img1 - img1_mean) * (img2 - img2_mean))
    denominator = np.sqrt(np.sum((img1 - img1_mean)**2) * np.sum((img2 - img2_mean)**2))
    if denominator == 0:
        return 0.0
    return numerator / denominator

def main():
    print("--- 2D-CS OSH Encryption Simulation ---")
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    os.makedirs('data/input', exist_ok=True)
    
    # 1. Load or Generate Test Images
    L_images = len(DISTANCES)
    plaintexts = []
    
    if HAS_SKIMAGE:
        from skimage.io import imread
        from skimage import data
        print("Loading specific benchmark images...")
        try:
            barbara = imread("data/input/barbara.png")
        except:
            barbara = data.camera() # Fallback for barbara
            
        images = [
            imread("data/input/parrot.png"),
            barbara,
            imread("data/input/house.png"),
            imread("data/input/boat.png")
        ]
        for img in images:
            img_r = resize(img, (N_PIXELS, N_PIXELS), anti_aliasing=True)
            if img_r.ndim == 3:
                img_r = np.mean(img_r, axis=2) # convert to grayscale
            img_r = img_r / np.max(img_r)
            plaintexts.append(img_r)
        
        while len(plaintexts) < L_images:
            plaintexts.append(np.fliplr(plaintexts[-1]))
        plaintexts = plaintexts[:L_images]
    else:
        print("Skimage not found. Generating dummy test images...")
        x = np.linspace(-1, 1, N_PIXELS)
        X, Y = np.meshgrid(x, x)
        dummy_imgs = [
            (np.sin(10 * X) + np.cos(10 * Y) + 2) / 4.0,
            np.exp(-(X**2 + Y**2) * 5),
            (np.sin(20 * X) + 1) / 2.0,
            (np.cos(20 * Y) + 1) / 2.0
        ]
        plaintexts.extend(dummy_imgs[:L_images])
        while len(plaintexts) < L_images:
            plaintexts.append(np.fliplr(plaintexts[-1]))
        
    for i, pt in enumerate(plaintexts):
        plt.imsave(f'results/plaintext_{i}.png', pt, cmap='gray')

    # 2. Matrix Generation
    print("Generating SPE Matrices...")
    keys, Qa = generate_spe_matrices(N_PIXELS, M_MEASUREMENTS, L_images)
    
    # 3. OSH Encryption
    print("Encrypting images...")
    ciphertext_I = OSH_encryption_eq21(plaintexts, keys, DISTANCES, WAVELENGTH, PIXEL_SIZE)
    
    plt.imsave('results/ciphertext_mag.png', np.abs(ciphertext_I), cmap='viridis')
    plt.imsave('results/ciphertext_real.png', np.real(ciphertext_I), cmap='gray')
    plt.imsave('results/ciphertext_imag.png', np.imag(ciphertext_I), cmap='gray')
    np.save('results/results_ciphertext.npy', ciphertext_I) # save full complex array
    np.save('results/ciphertext.npy', ciphertext_I)
    
    # 4. Decryption for each plaintext
    print("Starting decryption process...")
    for l in range(L_images):
        print(f"  Decrypting image {l} at z = {DISTANCES[l]}m...")
        Gamma_a, Phi_a, Phi_b = keys[l]['decryption_keys']
        
        # 4a. Trapdoor Extraction
        I_h = trapdoor_extraction_eq33(ciphertext_I, Gamma_a)
        
        # 4b. Image-Domain TV FISTA Solver
        print("    Running Image-Domain TV FISTA solver...")
        recovered_img = solve_2dcs_twist(
            I_h, Phi_a, Phi_b, N_PIXELS, WAVELENGTH, DISTANCES[l], PIXEL_SIZE, 
            iterations=ITERATIONS, tolerance=TOLERANCE
        )
        
        plt.imsave(f'results/recovered_{l}.png', recovered_img, cmap='gray')
        
        cc_score = calculate_cc(plaintexts[l], recovered_img)
        print(f"    => Correlation Coefficient for Image {l}: {cc_score:.4f}")

    print("Simulation Complete. Check the 'results' folder.")

if __name__ == "__main__":
    main()
