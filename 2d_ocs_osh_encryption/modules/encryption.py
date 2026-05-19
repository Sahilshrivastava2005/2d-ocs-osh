import numpy as np
import scipy.fft as fft
from modules.optics import fresnel_otf

def OSH_encryption_eq21(plaintexts, keys, distances, wavelength=632.8e-9, pixel_size=10e-6):
    """ Implements Eq. 21 and the optical diffraction components of Eq. 1 & 3. """
    N = plaintexts[0].shape[0]
    Qa = keys[0]['decryption_keys'][0].shape[0]
    sqrt_M = keys[0]['decryption_keys'][2].shape[1]
    ciphertext_I = np.zeros((Qa, sqrt_M), dtype=complex)
    
    for l, f_l in enumerate(plaintexts):
        z_l = distances[l]
        Gamma_a, Phi_a, Phi_b = keys[l]['decryption_keys']
        
        # Circular convolution (No pad, no crop) using frequency-domain OTF
        H_z = fresnel_otf(N, wavelength, z_l, pixel_size)
        F_f_l = fft.fft2(f_l, norm="ortho")
        F_l = fft.ifft2(F_f_l * H_z, norm="ortho")
        
        left_term = np.dot(Gamma_a, Phi_a)
        middle_term = np.dot(left_term, F_l)
        I_l = np.dot(middle_term, Phi_b)
        
        ciphertext_I += I_l
        
    return ciphertext_I