import numpy as np

# Units
Mpc				  = 3.08568025e22			# m
eV				  = 1.60217646e-19			# Electron volt


# Physical constants
c				  = 2.99792458e8			# m s-1
G				  = 6.67258e-11				# m3 kg-1 s-2
epsilon_0		  = 13.605698 * eV			# Ionization energy for H I
m_e				  = 9.10938188e-31			# Mass of electron [kg]
m_H				  = 1.673534e-27			# Mass of H [kg]
sigma_T			  =	6.652462e-29			# Thomson cross-section [m2]
alpha			  = 7.29735308e-3			# Not sure what this is. Ask.
hbar			  = 1.05457148e-34			# Planck constant [Js]
k_b				  = 1.3806503e-23			# Boltzmann constant [J/K]

# Cosmological parameters
Omega_b			  = 0.046
Omega_m			  = 0.224
Omega_r			  = 8.3e-5
Omega_nu		  = 0.0
Omega_lambda	  = 1. - Omega_m - Omega_b - Omega_r - Omega_nu
h0				  = 0.7
H_0				  = h0 * 100 * 1e3 / Mpc
rho_c0			  = 3 * H_0*H_0 / (8 * np.pi * G)
T_0				  = 2.725					# K
n_s				  = 1.0						# Spectral index (ask about this)
A_s				  =	1.0						# Don't know what this is. Ask



