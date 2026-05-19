import numpy as np
from scipy.linalg import hadamard

def generate_spe_matrices(N_pixels, M_measurements, L_images):
    """ Implements Eq. 22, 25, 31, and 32. """
    sqrt_M = int(np.sqrt(M_measurements))
    sqrt_N = N_pixels 
    Qa = L_images * sqrt_M 
    
    H_order = int(2**np.ceil(np.log2(Qa)))
    H_matrix = hadamard(H_order)
    
    # Permute rows to randomize spatial patterns and eliminate grid-like structure
    # while preserving column orthogonality
    rng = np.random.default_rng(42)
    H_matrix = H_matrix[rng.permutation(H_order), :]
    
    keys = []
    for l in range(L_images):
        Phi_a = np.random.normal(0, 1/N_pixels, (sqrt_M, sqrt_N))
        Phi_b = np.random.normal(0, 1/N_pixels, (sqrt_N, sqrt_M))
        Gamma_a = H_matrix[:Qa, (l * sqrt_M):((l + 1) * sqrt_M)]
        
        keys.append({
            'encryption_key': None, # Omitted to prevent OOM. Use properties of Kronecker product during encryption instead.
            'decryption_keys': (Gamma_a, Phi_a, Phi_b) 
        })
    return keys, Qa

def generate_structural_patterns(keys, N_pixels, Qa, M_measurements):
    """ Implements Eq. 29 and 30. """
    sqrt_M = int(np.sqrt(M_measurements))
    W = Qa * sqrt_M 
    L = len(keys)
    
    structural_patterns = np.zeros((W, N_pixels, N_pixels))
    
    for l in range(L):
        Delta_prime = keys[l]['encryption_key']
        for w in range(W):
            row_vector = Delta_prime[w, :]
            t_wl = row_vector.reshape((N_pixels, N_pixels))
            structural_patterns[w] += t_wl
            
    return structural_patterns