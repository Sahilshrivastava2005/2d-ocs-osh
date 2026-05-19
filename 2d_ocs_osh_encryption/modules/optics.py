import numpy as np

def fresnel_psf(N, wavelength, z, pixel_size):
    """ Calculates the Fresnel Point Spread Function h(x,y,z_l) from Eq. 1. """
    x = np.linspace(-N/2, N/2, N) * pixel_size
    y = np.linspace(-N/2, N/2, N) * pixel_size
    X, Y = np.meshgrid(x, y)
    
    k = 2 * np.pi / wavelength
    phase = (k / (2 * z)) * (X**2 + Y**2)
    h = np.exp(1j * phase) / (1j * wavelength * z)
    return h

def fresnel_otf(N, wavelength, z, pixel_size):
    """ Calculates the frequency-domain Optical Transfer Function (OTF) of Fresnel diffraction. """
    fx = np.fft.fftfreq(N, d=pixel_size)
    FX, FY = np.meshgrid(fx, fx)
    H_z = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    return H_z