import numpy as np

# Units
Mpc = 3.08568025e22
m2Mpc = 1./Mpc # Conversion from m to Mpc

# Cosmological parameters
Omega_b = 0.046
Omega_m = 0.224
Omega_r = 8.3e-5
Omega_nu = 0.0
Omega_lambda = 1. - Omega_m - Omega_b - Omega_r - Omega_nu
h0 = 0.7
H_0 = h0 * 100 * 1e3 / Mpc

# Physical constants
c = 2.99792458e8	# m s-1
