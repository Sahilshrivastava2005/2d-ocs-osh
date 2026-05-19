import numpy as np
import scipy.fft as fft
from skimage.restoration import denoise_tv_chambolle
from modules.optics import fresnel_otf

def trapdoor_extraction_eq33(ciphertext_I, Gamma_a_h):
    """ Implements Eq. 33 to extract sub-ciphertext. """
    I_h = np.dot(Gamma_a_h.T, ciphertext_I)
    H_order = int(2**np.ceil(np.log2(Gamma_a_h.shape[0])))
    return I_h / H_order

def solve_2dcs_twist(I_h, Phi_a, Phi_b, N_pixels, wavelength, z, pixel_size, iterations=400, tolerance=1e-3):
    """ Forward-Backward Image-Domain TV-FISTA Solver using Circular Convolution and Frequency OTF. """
    # Pre-compute optics in the frequency domain
    H_z = fresnel_otf(N_pixels, wavelength, z, pixel_size)
    H_z_conj = np.conj(H_z)
    
    # Power iteration to find exact Lipschitz constant of the full A*A operator
    v = np.random.randn(N_pixels, N_pixels)
    L = 1.0
    for _ in range(5):
        # Forward A(v)
        F_v_l = fft.ifft2(fft.fft2(v, norm="ortho") * H_z, norm="ortho")
        res = np.dot(np.dot(Phi_a, F_v_l), Phi_b)
        
        # Adjoint A*(res)
        grad_F = np.dot(np.dot(Phi_a.T, res), Phi_b.T)
        grad_f = np.real(fft.ifft2(fft.fft2(grad_F, norm="ortho") * H_z_conj, norm="ortho"))
        
        L = np.linalg.norm(grad_f)
        v = grad_f / L
        
    alpha = 1.0 / L
    
    f_est = np.zeros((N_pixels, N_pixels))
    Y = f_est.copy()
    t_k = 1.0
    
    for i in range(iterations):
        f_old = f_est.copy()
        
        # 1. Forward Optics on Y
        F_Y_l = fft.ifft2(fft.fft2(Y, norm="ortho") * H_z, norm="ortho")
        
        # 2. Measurement Residual
        residual = np.dot(np.dot(Phi_a, F_Y_l), Phi_b) - I_h
        
        if i % 100 == 0:
            print(f"    Iter {i}: Residual Norm = {np.linalg.norm(residual):.4f}")
        
        # 3. Adjoint Measurement
        grad_F = np.dot(np.dot(Phi_a.T, residual), Phi_b.T)
        
        # 4. Adjoint Optics
        grad_f = np.real(fft.ifft2(fft.fft2(grad_F, norm="ortho") * H_z_conj, norm="ortho"))
        
        # 5. Gradient Step
        temp = Y - alpha * grad_f
        
        # 6. TV Proximal Step (Sparsity prior on image domain)
        optimal_tv_weight = tolerance * 10.0
        f_est = denoise_tv_chambolle(temp, weight=optimal_tv_weight)
        
        # 7. Non-negativity constraint
        f_est = np.maximum(f_est, 0)
        
        # 8. Momentum Step (FISTA)
        t_k_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
        Y = f_est + ((t_k - 1.0) / t_k_next) * (f_est - f_old)
        t_k = t_k_next
        
    f_est = (f_est - np.min(f_est)) / (np.max(f_est) - np.min(f_est) + 1e-12)
    return f_est