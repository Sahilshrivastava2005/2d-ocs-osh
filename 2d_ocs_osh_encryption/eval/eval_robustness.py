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

def get_base_data():
    L_images = 4
    plaintexts = []
    
    if HAS_SKIMAGE:
        images = [data.camera(), data.astronaut(), data.moon(), data.coins()]
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
            
    keys, Qa = generate_spe_matrices(N_PIXELS, M_MEASUREMENTS, L_images)
    ciphertext_I = OSH_encryption_eq21(plaintexts, keys, DISTANCES, WAVELENGTH, PIXEL_SIZE)
    return plaintexts, keys, ciphertext_I

def run_robustness(noise_levels, occlusion_levels):
    os.makedirs('results/eval', exist_ok=True)
    plaintexts, keys, base_ciphertext = get_base_data()
    l = 0
    Gamma_a, Phi_a, Phi_b = keys[l]['decryption_keys']
    
    # 1. Figure 12: AWGN
    print("Running AWGN tests...")
    cc_snr = []
    signal_power = np.mean(np.abs(base_ciphertext)**2)
    
    for snr_db in noise_levels:
        snr_linear = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), base_ciphertext.shape)
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), base_ciphertext.shape)
        noisy_ciphertext = base_ciphertext + (noise_real + 1j * noise_imag)
        
        I_h = trapdoor_extraction_eq33(noisy_ciphertext, Gamma_a)
        recovered_img = solve_2dcs_twist(
            I_h, Phi_a, Phi_b, N_PIXELS, WAVELENGTH, DISTANCES[l], PIXEL_SIZE, 
            iterations=ITERATIONS, tolerance=TOLERANCE
        )
        
        cc = calculate_cc(plaintexts[l], recovered_img)
        cc_snr.append(cc)
        print(f"  SNR = {snr_db}dB -> CC = {cc:.4f}")
        
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, cc_snr, 'o-')
    plt.title("Correlation Coefficient vs SNR (Fig 12)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Correlation Coefficient (CC)")
    plt.grid(True)
    plt.savefig('results/eval/fig12_cc_vs_snr.png')
    print("Saved Figure 12.")

    # 2. Figure 13: Occlusion
    print("Running Occlusion tests...")
    cc_occ = []
    
    for occ_percent in occlusion_levels:
        occ_ciphertext = base_ciphertext.copy()
        
        total_elements = occ_ciphertext.size
        elements_to_zero = int(total_elements * (occ_percent / 100.0))
        
        side = int(np.sqrt(elements_to_zero))
        start_r = (occ_ciphertext.shape[0] - side) // 2
        start_c = (occ_ciphertext.shape[1] - side) // 2
        
        occ_ciphertext[start_r:start_r+side, start_c:start_c+side] = 0
        
        I_h = trapdoor_extraction_eq33(occ_ciphertext, Gamma_a)
        recovered_img = solve_2dcs_twist(
            I_h, Phi_a, Phi_b, N_PIXELS, WAVELENGTH, DISTANCES[l], PIXEL_SIZE, 
            iterations=ITERATIONS, tolerance=TOLERANCE
        )
        
        cc = calculate_cc(plaintexts[l], recovered_img)
        cc_occ.append(cc)
        print(f"  Occlusion = {occ_percent}% -> CC = {cc:.4f}")
        
    plt.figure(figsize=(8, 6))
    plt.plot(occlusion_levels, cc_occ, 's-', color='orange')
    plt.title("Correlation Coefficient vs Occlusion (Fig 13)")
    plt.xlabel("Occlusion Ratio (%)")
    plt.ylabel("Correlation Coefficient (CC)")
    plt.grid(True)
    plt.savefig('results/eval/fig13_cc_vs_occlusion.png')
    print("Saved Figure 13.")

if __name__ == "__main__":
    noise_levels = [5, 10, 15, 20, 25, 30] # dB
    occlusion_levels = [10, 20, 30, 40, 50, 60] # %
    run_robustness(noise_levels, occlusion_levels)
