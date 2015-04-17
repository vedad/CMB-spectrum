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
from CMBSpectrum import get_H, get_H_scaled, write2file

def get_n_p(x):
	"""
	Calculates the proton density n_b.
	----------------------------------
	x			: float, array
	returns	  	: float, array
	----------------------------------
	"""
	a = np.exp(x)

	return (params.Omega_b * params.rho_c0) / (params.m_H * a**3)

def get_T_b(x):
	"""
	Calculates the baryon temperature of the universe.
	--------------------------------------------------
	x			: float, array
	returns	  	: float, array
	--------------------------------------------------
	"""
	a = np.exp(x)

	return params.T_0 / a


def get_saha(x):
	"""
	Calculates the fractional electron density from Saha's equation.
	----------------------------------------------------------------
	x			: float, array
	returns	  	: float, array
	----------------------------------------------------------------
	"""
	n_b = get_n_p(x)
	T_b = get_T_b(x)

	a = 1.
	b = 1./n_b * (params.m_e * params.k_b * T_b /\
			(2 * np.pi * params.hbar*params.hbar))**1.5 *\
			np.exp(-params.epsilon_0 / (params.k_b * T_b))
	c = -b
	
	return (-b + np.sqrt(b*b - 4*a*c))/(2*a) # Other solution is always negative

def get_peebles(X_e, x):
	"""
	Calculates the fractional electron density from Peebles' equation.
	------------------------------------------------------------------
	X_e			: float
	x			: float, array
	returns	  	: float, array
	------------------------------------------------------------------
	"""
	n_b = get_n_p(x)
	T_b = get_T_b(x)
	H	= get_H(x)

	n_1s		  = (1 - X_e) * n_b
	Lambda_2s_1s  = 8.227
	Lambda_alpha  = H * (3 * params.epsilon_0)**3. /\
					(64 * np.pi * np.pi * (params.c * params.hbar)**3. *\
					n_1s)
	phi2		  = 0.448 * np.log(params.epsilon_0 / (params.k_b * T_b))
	alpha2		  = 64 * np.pi / np.sqrt(27 * np.pi) * params.alpha *\
					params.alpha / (params.m_e * params.m_e) *\
					params.hbar * params.hbar / params.c *\
					np.sqrt(params.epsilon_0 / (params.k_b * T_b)) * phi2
	beta		  = alpha2 * (params.m_e * params.k_b * T_b / (2 * np.pi *\
					params.hbar * params.hbar))**1.5 *\
					np.exp(-params.epsilon_0 / (params.k_b * T_b))
	beta2		  = alpha2 * (params.m_e * params.k_b * T_b / (2 * np.pi *\
					params.hbar * params.hbar))**1.5 *\
					np.exp(-params.epsilon_0 / (4 * params.k_b * T_b))

	if x > -6.35:
		C_r	= 1.
	else:
		C_r	= (Lambda_2s_1s + Lambda_alpha) /\
				(Lambda_2s_1s + Lambda_alpha + beta2)

	return C_r / H * (beta * (1 - X_e) - n_b * alpha2 * X_e * X_e)

def tau_rhs(tau, x):
	"""
	Right-hand side of differential equation of optical depth.
	Needed for odeint (ODE solver).
	----------------------------------------------------------
	tau			: float
	x			: float, array
	returns	  	: float, array
	----------------------------------------------------------
	"""
	return - get_n_e(x) * params.sigma_T * params.c / get_H(x)

def get_n_e(x):
	"""
	Calculates the electron density at arbitrary time x.
	----------------------------------------------------
	x			: float, array
	returns	  	: float, array
	----------------------------------------------------

	"""
	# Note:	ext is a keyword argument to specify what to return when
	#		the input x is outside of the precomputed grid. The number
	#		3 is an integer code to return the boundary value.
	return np.exp(splev(x, tck_n_e, der=0, ext=3))

def get_tau(x):
	"""
	Calculates the optical depth at arbitrary time x.
	-------------------------------------------------
	x			: float, array
	returns	  	: float, array
	-------------------------------------------------
	"""
	return np.exp(splev(x, tck_tau, der=0, ext=3))

def get_dtau(x):
	"""
	Calculates the first derivative of the optical depth
	at arbitrary time x.
	-----------------------------------------------------
	x			: float, array
	returns	  	: float, array
	-----------------------------------------------------
	"""
	return get_tau(x) * splev(x, tck_tau, der=1, ext=3)

def get_ddtau(x):
	"""
	Calculates the second derivative of the optical depth at
	arbitrary time x.
	--------------------------------------------------------
	x			: float, array
	returns	  	: float, array
	--------------------------------------------------------
	"""
	return get_tau(x) * splev(x, tck_tau, der=2, ext=3) + get_dtau(x) *\
			get_dtau(x) / get_tau(x)

def get_g(x):
	"""
	Computes the visibility function at arbitrary time x.
	-----------------------------------------------------
	x			: float, array
	returns	  	: float, array
	-----------------------------------------------------
	"""
	return splev(x, tck_g, der=0, ext=3)

def get_dg(x):
	"""
	Computes the first derivative of the visibility function
	at arbitrary time x.
	--------------------------------------------------------
	x			: float, array
	returns	  	: float, array
	--------------------------------------------------------
	"""
	return splev(x, tck_g, der=1, ext=3)

def get_ddg(x):
	"""
	Computes the second derivative of the visibility function
	at arbitrary time x.
	---------------------------------------------------------
	x			: float, array
	returns	  	: float, array
	---------------------------------------------------------
	"""
	# Note: splev returns continuous values for derivatives by specifying
	#		der=1 for first derivative, der=2 for second derivative.
	#		No need to spline the discrete double derivatives as mentioned
	#		in the exercise text.
	return splev(x, tck_g, der=2, ext=3)

### Initialization

start = time.time()

saha_limit	= 0.99			 # Switch from Saha to Peebles when X_e > 0.99
xstart	    = np.log(1e-10)	 # Start grids at a = 10^(-10)
xstop		= 0.0			 # Stop grids at a = 1
n			= 1000			 # Number of grid points between xstart and xstop

X_e			= np.zeros(n)
tau			= np.zeros(n)

x_rec		= np.linspace(xstart, xstop, n)

# Find electron fraction > 0.99 by solving Saha equation
X_e[0]		= get_saha(x_rec[0])
X_e_val		= X_e[0]
i = 1

while X_e_val > 0.99:
	X_e[i]	= get_saha(x_rec[i])
	X_e_val	= X_e[i]
 	i += 1

# Find rest of electron fraction by solving Peebles' equation
X_e[i-2:] = odeint(get_peebles, X_e[i-2], x_rec[i-2:])[:,0]

# Compute electron density
n_e		  = X_e * get_n_p(x_rec)

# Spline log of electron density (smoother spline)
tck_n_e	  = splrep(x_rec, np.log(n_e))

# Compute optical depth
tau[0]	  = 0.0
tau		  = odeint(tau_rhs, tau[0], x_rec[::-1])[:,0][::-1]

# Spline tau
tck_tau	  = splrep(x_rec[:-1], np.log(tau[:-1]))

# Calculate g for spline
g		  =	- get_dtau(x_rec) * np.exp(-get_tau(x_rec))

# Spline g
tck_g	  = splrep(x_rec, g)
	
stop = time.time()

if __name__ == "__main__":

	print "Change to Peebles' equation at z = %g" % (1./np.exp(x_rec[i-2]) - 1)
	print "Runtime: %g seconds." % (stop - start)
	
	# Testing splined functions
	X_e_test	= get_n_e(x_rec) / get_n_p(x_rec)

	tau_test	= get_tau(x_rec)
	dtau_test	= get_dtau(x_rec)
	ddtau_test	= get_ddtau(x_rec)

	g_test		= get_g(x_rec)
	dg_test		= get_dg(x_rec)
	ddg_test	= get_ddg(x_rec)

	### Writing results to files

	# Results from electron fraction
	write2file("../data/milestone2/electron_fraction.txt",\
			"Evolution of the electron fraction as function of redshift: x\
			X_e", x_rec, X_e_test)

	# Results from optical depth
	write2file("../data/milestone2/optical_depth.txt",\
			"Optical depth, its first and second derivatives\
			as function of time: x tau dtau ddtau", x_rec, tau_test,\
			dtau_test, ddtau_test)

	# Results from visibility function
	write2file("../data/milestone2/visibility_function.txt",\
			"Visibility function and its first and second derivatives\
			as function of time: x g dg ddg", x_rec, g_test, dg_test,\
			ddg_test)
