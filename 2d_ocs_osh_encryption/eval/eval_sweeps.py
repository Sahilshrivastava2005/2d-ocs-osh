import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_PIXELS, WAVELENGTH, PIXEL_SIZE, TOLERANCE, DISTANCES
from modules.matrix_gen import generate_spe_matrices
from modules.encryption import OSH_encryption_eq21
from modules.decryption import trapdoor_extraction_eq33, solve_2dcs_twist

def calculate_cc(img1, img2):
    img1_mean = np.mean(img1)
    img2_mean = np.mean(img2)
    numerator = np.sum((img1 - img1_mean) * (img2 - img2_mean))
    denominator = np.sqrt(np.sum((img1 - img1_mean)**2) * np.sum((img2 - img2_mean)**2))
    if denominator == 0:
        return 0.0
    return numerator / denominator

def generate_plaintexts():
    plaintexts = []
    x = np.linspace(-1, 1, N_PIXELS)
    X, Y = np.meshgrid(x, x)
    dummy_imgs = [
        (np.sin(10 * X) + np.cos(10 * Y) + 2) / 4.0,
        np.exp(-(X**2 + Y**2) * 5),
        (np.sin(20 * X) + 1) / 2.0,
        (np.cos(20 * Y) + 1) / 2.0
    ]
    plaintexts.extend(dummy_imgs)
    return plaintexts

def run_sweep(M_measurements, iterations_list):
    L_images = 4
    plaintexts = generate_plaintexts()
    
    print(f"--- Running Sweep for M = {M_measurements} ---")
    keys, Qa = generate_spe_matrices(N_PIXELS, M_measurements, L_images)
    ciphertext_I = OSH_encryption_eq21(plaintexts, keys, DISTANCES, WAVELENGTH, PIXEL_SIZE)
    
    # We will test only image 0
    l = 0
    Gamma_a, Phi_a, Phi_b = keys[l]['decryption_keys']
    I_h = trapdoor_extraction_eq33(ciphertext_I, Gamma_a)
    
    cc_scores = []
    
    for iters in iterations_list:
        print(f"  Testing Iterations = {iters}...")
        recovered_img = solve_2dcs_twist(
            I_h, Phi_a, Phi_b, N_PIXELS, WAVELENGTH, DISTANCES[l], PIXEL_SIZE, 
            iterations=iters, tolerance=TOLERANCE
        )
        cc_score = calculate_cc(plaintexts[l], recovered_img)
        cc_scores.append(cc_score)
        
    return cc_scores

def eval_sweeps():
    os.makedirs('results/eval', exist_ok=True)
    
    iterations_list = [10, 50, 100, 200, 300, 400]
    
    print("Testing 14% Sampling Rate (M=9216)...")
    cc_14 = run_sweep(9216, iterations_list)
    
    print("Testing 6% Sampling Rate (M=3969)...")
    cc_6 = run_sweep(3969, iterations_list)
    
    plt.figure(figsize=(8, 6))
    plt.plot(iterations_list, cc_14, 'o-', label='14% Sampling Rate')
    plt.plot(iterations_list, cc_6, 's-', label='6% Sampling Rate')
    plt.title("Correlation Coefficient vs Iterations (Fig 11)")
    plt.xlabel("Iterations")
    plt.ylabel("Correlation Coefficient (CC)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig('results/eval/fig11_cc_vs_iterations.png')
    print("Saved Figure 11 to results/eval/fig11_cc_vs_iterations.png")

if __name__ == "__main__":
    eval_sweeps()
