#!/usr/bin/env python

"""
Created on 19 Jan. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt
import params
from scipy.integrate import odeint
from scipy.interpolate import splrep

#def __init__(self, n_t, x_t, a_t, n_eta, x_eta, deta):
#	"""
#	An object (CMB spectrum) has attributes:
#	n_t	  :	  Number of x-values
#	x_t	  :	  Grid of x-values
#	a_t	  :	  Grid of a-values (scale factor)
#
#	n_eta :	  Number of grid points in conformal time
#	x_eta :	  Grid points in conformal time
#	eta	  :	  Conformal time at each grid point
#	eta2  :	  Spline array of eta points (I think?)
#	"""
#	self.n_t, self.x_t, self.a_t, self.n_eta, self.x_eta, self.eta, self.eta2 \
#			= n_t, x_t, a_t, n_eta, x_eta, eta, eta2

def get_H(x):
	"""
	Computes the Hubble parameter H at given x.
	"""
	a = np.exp(x)
#	return params.H_0 * np.sqrt(params.Omega_M * a**(-3))
	return params.H_0 * np.sqrt((params.Omega_m + params.Omega_b) * a**(-3) +\
			params.Omega_r * a**(-4) + params.Omega_lambda)

def get_H_scaled(x):
	"""
	Computes the scaled Hubble parameter H' = a*H at given x.
	"""
	a = np.exp(x)
#	return params.H_0 * np.sqrt((params.Omega_M / a))
	return params.H_0 * np.sqrt((params.Omega_m + params.Omega_b) / a +\
			params.Omega_r / (a * a) + params.Omega_lambda * a * a)

def get_dH_scaled(x):
	return None

def get_eta(x, tck):
	"""
	Computes eta(x) using the previously computed spline function.
	"""
	return splev(x, tck, der=0)

def eta_rhs(eta, x):
	"""
	Solves the differential equation d eta/da = c / (a * H_p)
	"""
	a = np.exp(x)
	rhs = (params.c * params.m2Mpc) / (a * get_H_scaled(x))
	return rhs

def write2file(filename, header, a, b):
	outFile = open(filename, 'w')
	outFile.write("# " + header + "\n")

	for i in xrange(len(a)):
		outFile.write('%.12f %.12f\n' % (a[i], b[i]))
	  
	outFile.close()

	
if __name__ == "__main__":

	### Initialization

	# Define two epochs: 1) during and 2) after recombination
	n1			= 200						# Number of grid points during recombination
	n2			= 300						# Number of grid points after recombination
	n_t			= n1 + n2					# Total number of grid points

	z_start_rec = 1630.4					# Redshift at start of recombination
	z_end_rec	= 614.2					    # Redshift at end of recombination
	z_0			= 0							# Redshift today

	x_start_rec = -np.log(1 + z_start_rec)	# x at start of recombination
	x_end_rec	= -np.log(1 + z_end_rec)	# x at end of recombination
	x_0			= 0							# x today

	n_eta		= 1000						# Number of eta grid points (for spline)
	a_init		= 1e-10						# Start value of a for eta evaluation
	x_eta1		= np.log(a_init)			# Start value of x for eta evaluation
	x_eta2		= 0							# End value of x for eta evaluation

	# Grid for x
	x1 = np.linspace(int(x_start_rec), int(x_end_rec), n1)
	x2 = np.linspace(int(x_end_rec), int(x_0), n2)
	x_t = np.concatenate((x1, x2), axis=0)	# Concatenates two arrays

	# Grid for a
	a_t = np.exp(x_t) # Since x = ln(a)

	# Grid for x in conformal time
	x_eta = np.linspace(int(x_eta1), int(x_eta2), n_eta)
	a_eta = np.exp(x_eta)

	# a*a*H -> H_0 * sqrt(Omega_r) as a -> 0 (which is valid for a = 1e-10)
	eta0 = (params.c * params.m2Mpc) / (params.H_0 * np.sqrt(params.Omega_r))

	# Solve the differential equation for eta
	eta = odeint(eta_rhs, eta0, x_eta)

	write2file("../data/conformal_time.txt", "Conformal time values: a_eta eta",
			a_eta, eta)

	write2file("../data/hubble_constant.txt", "Hubble constant values: H a",
			a_eta, (params.Mpc / 1e3) * get_H(x_eta))

	tck = splrep(a_eta, eta, s=0)
#	eta_ipl = splev(x_spl, tck, der=0)


#	write2file("../data/conformal_time_interpolate.txt",\
#			"Interpolated conformal time values: eta_ipl

