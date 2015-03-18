#!/usr/bin/env python

"""
Created on 16 Mar. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np							  # Functions, mathematical operators
import matplotlib.pyplot as plt				  # Plotting
from scipy.integrate import odeint			  # ODE solver
from scipy.interpolate import splrep, splev	  # Spline interpolation
import time									  # Time module for runtime
import params								  # Module for physical constants and parameters
import CMBSpectrum

def get_n_p(x):
	"""
	Calculates the proton density n_b.
	"""
	a = np.exp(x)

#	return (CMBSpectrum.get_Omega_b(x) * params.rho_c0) / (params.m_H * a**3)
	return (params.Omega_b * params.rho_c0) / (params.m_H * a**3)

def get_T_b(x):
	"""
	Calculates the baryon temperature of the universe.
	Assumes T_b = T_r = T_0 / a
	"""
	a = np.exp(x)

	return params.T_0 / a


def get_saha(x):
	"""
	Calculates the fractional electron density X_e = n_e/n_H
	from Saha's equation.
	"""
	n_b = get_n_p(x)
	T_b = get_T_b(x)

	a = 1.
	b = 1./n_b * (params.m_e * params.k_b * T_b /\
			(2 * np.pi * params.hbar*params.hbar))**1.5 *\
			np.exp(-params.epsilon_0 / (params.k_b * T_b))
	c = -b
	
	return (-b + np.sqrt(b*b - 4*a*c))/(2*a)	# Other solution is always negative

def get_peebles(X_e, x):
	"""
	Calculates the fractional electron density X_e from Peebles' equation.
	"""
	n_b = get_n_p(x)
	T_b = get_T_b(x)
	H	= CMBSpectrum.get_H(x)

	n_1s			= (1 - X_e) * n_b
	Lambda_2s_1s	= 8.227
	Lambda_alpha	= H * (3 * params.epsilon_0)**3. /\
						(64 * np.pi * np.pi * (params.c * params.hbar)**3. *\
						n_1s)
	phi2			= 0.448 * np.log(params.epsilon_0 / (params.k_b * T_b))
	alpha2			= 64 * np.pi / np.sqrt(27 * np.pi) * params.alpha *\
						params.alpha / (params.m_e * params.m_e) *\
						params.hbar * params.hbar / params.c *\
						np.sqrt(params.epsilon_0 / (params.k_b * T_b)) * phi2
	beta			= alpha2 * (params.m_e * params.k_b * T_b / (2 * np.pi *\
						params.hbar * params.hbar))**1.5 *\
						np.exp(-params.epsilon_0 / (params.k_b * T_b))
	beta2			= alpha2 * (params.m_e * params.k_b * T_b / (2 * np.pi *\
						params.hbar * params.hbar))**1.5 *\
						np.exp(-params.epsilon_0 / (4 * params.k_b * T_b))
	if x > -6.35:
		C_r			= 1.
	else:
		C_r			= (Lambda_2s_1s + Lambda_alpha) /\
						(Lambda_2s_1s + Lambda_alpha + beta2)

	return C_r / H * (beta * (1 - X_e) - n_b * alpha2 * X_e * X_e)

def tau_rhs(tau, x, n_e):
	"""
	Right-hand side of differential equation of optical depth.
	Needed for odeint (ODE solver).
	"""
	a = np.exp(x)

	# No negative sign because we solve the ODE from today and back in time (?)
	return n_e * params.sigma_T * a / CMBSpectrum.get_H_scaled(x)

def get_n_e(x, tck):
	"""
	Calculates the electron density at arbitrary time x from a splined grid.
	"""

	tck = splrep(x, np.log10(n_e))
	return 10**(splev(x, tck, der=0))

def get_tau(x, tck):
	"""
	Calculates the optical depth as function of x using previously splined grid.
	"""
	return 10**(splev(x, tck, der=0))	
	


if __name__ == "__main__":

	start = time.time()

	saha_limit	= 0.99					  # Switch from Saha to Peebles when X_e > 0.99
	xstart	    = np.log(1e-10)			  # Start grids at a = 10^(-10)
	xstop		= 0.0					  # Stop grids at a = 1
	n			= 1000					  # Number of grid points between xstart and xstop

	X_e			= np.zeros(n)
	tau			= np.zeros(n)
	tau2		= np.zeros(n)
	tau22		= np.zeros(n)
	n_e			= np.zeros(n)
	n_e2		= np.zeros(n)
	g			= np.zeros(n)
	g2			= np.zeros(n)
	g22			= np.zeros(n)

	# Grid for tau
	x_rec		= np.linspace(xstart, xstop, n)

	# Grid for electron fraction
	x_e			= np.linspace(np.log(1./(1850+1)), 0.0, n)
	
	X_e[0]		= get_saha(x_e[0])
	X_e_val		= X_e[0]
	i = 1

	print "Initialised with Saha equation."

	# Find electron fraction > 0.99 by solving Saha equation
	while X_e_val > 0.99:
		X_e[i]	= get_saha(x_e[i])
		X_e_val	= X_e[i]
 		i += 1

	print "Change to Peebles' equation at z = %g" % (1./np.exp(x_e[i-1]) - 1)
	
	"""
	print ((X_e[i-3] - X_e[i-4]) / (x_e[i-3] - x_e[i-4]))
	print ((X_e[i-2] - X_e[i-3]) / (x_e[i-2] - x_e[i-3]))	  # -0.31
	print ((X_e[i-1] - X_e[i-2]) / (x_e[i-1] - x_e[i-2]))
#	print get_peebles(X_e[i-2], x_e[i-2])
#	print get_peebles(X_e[i-1], x_e[i-1])					  # -5.1e-12
	i0 = i
	i1 = i-1
	i2 = i-2
	print X_e[i-4:i+3]
	"""

	# Find rest of electron fraction by solving Peebles' equation
	X_e[i-1:] = odeint(get_peebles, X_e[i-2], x_e[i-1:])[:,0]

	"""
	print get_peebles(X_e[i2-1], x_e[i2-1])
	print get_peebles(X_e[i2], x_e[i2])
	print get_peebles(X_e[i1], x_e[i1])
	print get_peebles(X_e[i0], x_e[i0])
	"""

	# Compute electron density
	n_e		= X_e * get_n_p(x_e)

	# Spline log of electron density (smoother spline)
	tck_n_e	= splrep(x_e, np.log10(n_e))

	# Compute optical depth
	tau0	= 0.0
	tau		= odeint(tau_rhs, tau0, x_rec, args=(n_e,))[:,0]
	print x_rec
#	print tau

	# Spline tau
	tck_tau	= splrep(x_rec, tau)

	CMBSpectrum.write2file("../data/milestone2/electron_fraction.txt",\
			"Evolution of the electron fraction as function of redshift: x\
			X_e", x_e, X_e)

	CMBSpectrum.write2file("../data/milestone2/optical_depth.txt",\
			"Optical depth as function of time: x tau", x_rec, tau)

	stop = time.time()

	print "Runtime: %g seconds." % (stop - start)
