#!/usr/bin/env python

"""
Created on 19 Jan. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np							  # Functions, mathematical operators
import matplotlib.pyplot as plt				  # Plotting
import params								  # Module for physical constants and parameters
from scipy.integrate import odeint			  # ODE solver
from scipy.interpolate import splrep, splev	  # Spline interpolation

def get_H(x):
	"""
	Computes the Hubble parameter H.
	--------------------------------
	x		  :	float, array
	returns	  :	float, array
	--------------------------------
	"""
	a = np.exp(x)

	return params.H_0 * np.sqrt((params.Omega_m + params.Omega_b) * a**(-3) +\
			params.Omega_r * a**(-4) + params.Omega_lambda)

def get_H_scaled(x):
	"""
	Computes the scaled Hubble parameter H' = a*H.
	----------------------------------------------
	x		  :	float, array
	returns	  :	float, array
	----------------------------------------------
	"""
	a = np.exp(x)

	return params.H_0 * np.sqrt((params.Omega_m + params.Omega_b) / a +\
			params.Omega_r / (a * a) + params.Omega_lambda * a * a)

def get_dH_scaled(x):
	"""
	Computes the derivative of H_p.
	-------------------------------
	x		  :	float, array
	returns	  : float, array
	-------------------------------
	"""
	a = np.exp(x)

	return H_0 * (-(params.Omega_b + params.Omega_m) / a -\
			2 * params.Omega_r / (a * a) + 2 * params.Omega_lambda * a * a) /\
			(2 * np.sqrt((params.Omega_b + params.Omega_m) / a +\
			params.Omega_r / (a * a) + params.Omega_lambda * a * a))

def get_eta(x):
	"""
	Computes eta(x) using tck from the previously computed spline function.
	-----------------------------------------------------------------------
	x		  : float, array
	tck		  : tuple
	returns	  :	float, array
	-----------------------------------------------------------------------
	"""
	a = np.exp(x)

	return splev(a, tck_eta, der=0)

def eta_rhs(eta, x):
	"""
	Solves the differential equation d eta/da = c / (a * H_p).
	(eta is needed as argument for the ODE solver)
	----------------------------------------------------------
	eta		  : float, array
	x		  : float, array
	returns	  :	float, array
	----------------------------------------------------------
	"""
	rhs = params.c / get_H_scaled(x)

	return rhs

def get_rho_c(x):
	"""
	Computes the critical density for given time.
	---------------------------------------------
	x		  :	float, array
	returns	  :	float, array
	---------------------------------------------
	"""
	return 3 * get_H(x)*get_H(x) / (8 * np.pi * params.G)

def get_Omega_m(x):
	"""
	Computes the time evolution of dark matter density.
	---------------------------------------------------
	x		  : float, array
	returns	  :	float, array
	---------------------------------------------------

	"""
	a = np.exp(x)

	rho_m0 = params.rho_c0 * params.Omega_m
	rho_m = rho_m0 / (a*a*a)

	return rho_m / get_rho_c(x)

def get_Omega_b(x):
	"""
	Computes the time evolution of baryon density.
	----------------------------------------------
	x		  : float, array
	returns	  :	float, array
	----------------------------------------------

	"""
	a = np.exp(x)

	rho_b0 = params.rho_c0 * params.Omega_b
	rho_b = rho_b0 / (a*a*a)
	
	return rho_b / get_rho_c(x)

def get_Omega_r(x):
	"""
	Computes the time evolution of radiation density.
	-------------------------------------------------
	x		  : float, array
	returns	  :	float, array
	-------------------------------------------------
	"""
	a = np.exp(x)

	rho_r0 = params.rho_c0 * params.Omega_r
	rho_r = rho_r0 / (a*a*a*a)

	return rho_r / get_rho_c(x)

def get_Omega_lambda(x):
	"""
	Computes the time evolution of the cosmological constant density.
	-----------------------------------------------------------------
	x		  : float, array
	returns	  :	float, array
	-----------------------------------------------------------------
	"""
	return 1. - get_Omega_m(x) - get_Omega_b(x) - get_Omega_r(x)

def write2file(filename, header, *args):
	"""
	Function that writes data (args) to specified filename.
	-------------------------------------------------------
	filename  :	string
	header	  :	string
	args	  :	array	  (optional)
	-------------------------------------------------------
	"""

	outFile = open(filename, 'w')
	outFile.write("# " + header + "\n")

	for i in xrange(len(args[0])):
		for arg in args:
			outFile.write('%.12f ' % arg[i])
		outFile.write('\n')

	outFile.close()


### Initialization

# Define two epochs: 1) during and 2) after recombination
n1			= 200						# Number of grid points during recombination
n2			= 300						# Number of grid points after recombination
n_t			= n1 + n2					# Total number of grid points

z_start_rec = 1630.4					# Redshift at start of recombination
z_end_rec	= 614.2					    # Redshift at end of recombination
z_0			= 0.						# Redshift today

x_start_rec	= -np.log(1. + z_start_rec)	# x at start of recombination
x_end_rec	= -np.log(1. + z_end_rec)	# x at end of recombination
x_0			= 0.						# x today

n_eta		= 1000						# Number of eta grid points (for spline)
a_init		= 1e-10						# Start value of a for eta evaluation
x_eta1		= np.log(a_init)			# Start value of x for eta evaluation
x_eta2		= 0.						# End value of x for eta evaluation

# Grid for x
x1			= np.linspace(x_start_rec, x_end_rec, n1)
x2			= np.linspace(x_end_rec, x_0, n2)
x_t			= np.concatenate((x1, x2), axis=0)

# Grid for a
a_t			= np.exp(x_t)				# Since x = ln(a)

# Grid for x in conformal time
x_eta		= np.linspace(x_eta1, x_eta2, n_eta)
a_eta		= np.exp(x_eta)

# Initial value for eta
eta0		= (params.c / (params.H_0 *\
			  np.sqrt(params.Omega_r))) * a_init

# Solve the differential equation for eta
eta			= odeint(eta_rhs, eta0, x_eta)
eta			= eta[:,0]

# Finding the vector of knots, B-spline  coefficients and degree of spline
tck_eta		= splrep(a_eta, eta)

if __name__ == "__main__":

	# Write conformal time data to file
	write2file("../data/milestone1/conformal_time.txt", "Conformal time values: a_eta eta",
			a_eta, x_eta, eta / params.Mpc)

	# Write Hubble constant data to file
	write2file("../data/milestone1/hubble_constant.txt", "Hubble constant values: a x H",
			a_eta, x_eta, (params.Mpc / 1e3) * get_H(x_eta))

	# Write density parameters data to file
	write2file("../data/milestone1/density_evolution.txt",\
			"Time evolution of cosmological parameters: a Omega_m Omega_b\
			Omega_r Omega_lambda", a_eta, x_eta, get_Omega_m(x_eta),\
			get_Omega_b(x_eta), get_Omega_r(x_eta), get_Omega_lambda(x_eta))
