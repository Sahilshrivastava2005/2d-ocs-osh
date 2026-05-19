import numpy as np
import scipy.fft as fft
from skimage.restoration import denoise_tv_chambolle
from skimage import data
from skimage.transform import resize
from modules.optics import fresnel_otf

N = 128
M = 4096
wavelength = 632.8e-9
pixel_size = 10e-6
z = 0.18

# 1. Ground truth image
img = resize(data.camera(), (N, N), anti_aliasing=True)
img = img / np.max(img)

# 2. Forward Optics
H_z = fresnel_otf(N, wavelength, z, pixel_size)
F_l = fft.ifft2(fft.fft2(img, norm="ortho") * H_z, norm="ortho")

# 3. Measurement
Phi_a = np.random.normal(0, 1/N, (int(np.sqrt(M)), N))
Phi_b = np.random.normal(0, 1/N, (N, int(np.sqrt(M))))
I_h = np.dot(np.dot(Phi_a, F_l), Phi_b)

# 4. Inverse FISTA with TV on image domain
v = np.random.randn(N, N)
H_z_conj = np.conj(H_z)
for _ in range(5):
    F_v_l = fft.ifft2(fft.fft2(v, norm="ortho") * H_z, norm="ortho")
    res = np.dot(np.dot(Phi_a, F_v_l), Phi_b)
    grad_F = np.dot(np.dot(Phi_a.T, res), Phi_b.T)
    grad_f = np.real(fft.ifft2(fft.fft2(grad_F, norm="ortho") * H_z_conj, norm="ortho"))
    L = np.linalg.norm(grad_f)
    v = grad_f / L

alpha = 1.0 / L
f_est = np.zeros((N, N))
Y = f_est.copy()
t_k = 1.0

for i in range(100):
    f_old = f_est.copy()
    
    # Forward Y
    F_Y_l = fft.ifft2(fft.fft2(Y, norm="ortho") * H_z, norm="ortho")
    
    # Residual
    res = np.dot(np.dot(Phi_a, F_Y_l), Phi_b) - I_h
    
    # Adjoint Measurement
    grad_F = np.dot(np.dot(Phi_a.T, res), Phi_b.T)
    
    # Adjoint Optics
    grad_f = np.real(fft.ifft2(fft.fft2(grad_F, norm="ortho") * H_z_conj, norm="ortho"))
    
    # Gradient step
    temp = Y - alpha * grad_f
    
    # TV Denoising (Proximal step)
    f_est = denoise_tv_chambolle(temp, weight=0.0001)
    f_est = np.maximum(f_est, 0)
    
    # Momentum
    t_k_next = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
    Y = f_est + ((t_k - 1) / t_k_next) * (f_est - f_old)
    t_k = t_k_next

# CC
cc = np.corrcoef(img.flatten(), f_est.flatten())[0, 1]
print(f"CC with Image Domain TV FISTA: {cc:.4f}")

