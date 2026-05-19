import numpy as np
from config import N_PIXELS, WAVELENGTH, PIXEL_SIZE, DISTANCES
from modules.matrix_gen import generate_spe_matrices
from modules.encryption import OSH_encryption_eq21
from modules.decryption import trapdoor_extraction_eq33, solve_2dcs_twist
from skimage import data
from skimage.transform import resize

M_MEASUREMENTS = 16384
TOLERANCE = 1e-5
ITERATIONS = 400

images = [data.camera(), data.astronaut(), data.moon(), data.coins()]
plaintexts = []
for img in images:
    img_r = resize(img, (N_PIXELS, N_PIXELS), anti_aliasing=True)
    if img_r.ndim == 3: img_r = np.mean(img_r, axis=2)
    plaintexts.append(img_r / np.max(img_r))

keys, Qa = generate_spe_matrices(N_PIXELS, M_MEASUREMENTS, 4)
ciphertext_I = OSH_encryption_eq21(plaintexts, keys, DISTANCES, WAVELENGTH, PIXEL_SIZE)

for l in range(4):
    Gamma_a, Phi_a, Phi_b = keys[l]['decryption_keys']
    I_h = trapdoor_extraction_eq33(ciphertext_I, Gamma_a)
    f_est = solve_2dcs_twist(I_h, Phi_a, Phi_b, N_PIXELS, WAVELENGTH, DISTANCES[l], PIXEL_SIZE, iterations=ITERATIONS, tolerance=TOLERANCE)
    cc = np.corrcoef(plaintexts[l].flatten(), f_est.flatten())[0, 1]
    print(f"Image {l} CC: {cc:.4f}")
