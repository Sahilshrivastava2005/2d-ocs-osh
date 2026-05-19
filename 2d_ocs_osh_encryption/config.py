import numpy as np

# Image dimensions
N_PIXELS = 256  # Target size of the image NxN

# 2D-CS Compression
# Note: At 25% sampling rate, the recovered image will naturally have some blur 
# (missing high frequencies). To make it perfectly sharp, increase M to 32768 (50%) or 65536 (100%).
# For N=256, total pixels = 65536. M = 16384 gives 25% sampling rate.
M_MEASUREMENTS = 16384

# Optics Parameters
WAVELENGTH = 632.8e-9  # He-Ne laser wavelength in meters
PIXEL_SIZE = 10e-6     # 10 micrometers
# Distance parameters for plaintexts (z_l)
DISTANCES = [0.18, 0.20, 0.22, 0.24]  # 4 plaintexts

# Solver hyperparameters (FISTA / TwIST)
ITERATIONS = 800
TOLERANCE = 1e-5
