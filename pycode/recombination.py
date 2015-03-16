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
import params								  # Module for physical constants and parameters
import CMBSpectrum

def get_proton_density(x):
	"""
	Calculates the proton density n_b.
	"""
	a = np.exp(x)

#	return (CMBSpectrum.get_Omega_b(x) * params.rho_c0) / (params.m_H * a**3)
	return (params.Omega_b * params.rho_c0) / (params.m_H * a**3)

def get_baryon_temperature(x):
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
	n_b = get_proton_density(x)
	T_b = get_baryon_temperature(x)

	a = 1.
	b = 1./n_b * ((2 * np.pi * params.m_e * params.k_b * T_b) /\
			(params.hbar*params.hbar))**(3./2) *\
			np.exp(params.epsilon_0 / ((params.k_b * T_b)/params.eV))
	c = -b
	
	return (-b + np.sqrt(b*b - 4*a*c))/(2*a)	# Other solution is always negative



if __name__ == "__main__":

	saha_limit	= 0.99					  # Switch from Saha to Peebls when X_e > 0.99
	xstart	    = np.log(1e-10)		  # Start grids at a = 10^(-10)
	xstop		= 0.0					  # Stop grids at a = 1
	n			= 1000					  # Number of grid points between xstart and xstop

	x_rec		= np.zeros(n)
	X_e			= np.zeros(n)
	tau			= np.zeros(n)
	tau2		= np.zeros(n)
	tau22		= np.zeros(n)
	n_e			= np.zeros(n)
	n_e2		= np.zeros(n)
	g			= np.zeros(n)
	g2			= np.zeros(n)
	g22			= np.zeros(n)

	x_rec		= np.linspace(xstart, xstop, n)
	

