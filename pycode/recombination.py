#!/usr/bin/env python

"""
Created on 16 Mar. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np							  # Functions, mathematical operators
import matplotlib.pyplot as plt				  # Plotting
from scipy.integrate import odeint			  # ODE solver
#from scipy.interpolate import splrep, splev	  # Spline interpolation
import params								  # Module for physical constants and parameters
import CMBSpectrum



if __name__ == "__main__":

	saha_limit	= 0.99					  # Switch from Saha to Peebls when X_e > 0.99
	xstart	    = np.log10(1e-10)		  # Start grids at a = 10^(-10)
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
	
	print len(x_rec)
	print x_rec

