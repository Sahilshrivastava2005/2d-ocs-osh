import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
import sys
import os

from config import N_PIXELS, WAVELENGTH, PIXEL_SIZE, DISTANCES, ITERATIONS, M_MEASUREMENTS
from modules.matrix_gen import generate_spe_matrices
from modules.encryption import OSH_encryption_eq21
from modules.decryption import trapdoor_extraction_eq33, solve_2dcs_twist

print("Generating data...")
L_images = 1
img = resize(data.camera(), (N_PIXELS, N_PIXELS), anti_aliasing=True)
plaintexts = [img / np.max(img)]

keys, Qa = generate_spe_matrices(N_PIXELS, M_MEASUREMENTS, L_images)
ciphertext_I = OSH_encryption_eq21(plaintexts, keys, [DISTANCES[0]], WAVELENGTH, PIXEL_SIZE)

Gamma_a, Phi_a, Phi_b = keys[0]['decryption_keys']
I_h = trapdoor_extraction_eq33(ciphertext_I, Gamma_a)

weights = [1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1]
for w in weights:
    print(f"Testing weight = {w}...")
    f_est = solve_2dcs_twist(I_h, Phi_a, Phi_b, N_PIXELS, WAVELENGTH, DISTANCES[0], PIXEL_SIZE, iterations=300, tolerance=w)
    plt.imsave(f'results/test_weight_{w}.png', f_est, cmap='gray')
    cc = np.corrcoef(plaintexts[0].flatten(), f_est.flatten())[0, 1]
    print(f"  CC = {cc:.4f}")

