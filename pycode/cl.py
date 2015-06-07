#!/usr/bin/env python

"""
Created on 13 May 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import splrep, splev
from scipy.special import jv
import time
from params import c, H_0, Omega_b, Omega_r, Omega_m, Omega_lambda, n_s
from CMBSpectrum import x_start_rec
from CMBSpectrum import get_eta, write2file
from evolution import S_hires, n_hires, k_hires, x_hires, k_min, k_max

start = time.time()

ls	  = np.array([2, 3, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100,\
				120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550,\
				600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200])
l_num = len(ls)

n_spline  = 5400
z_spline  = np.linspace(0,3500,n_spline)

# Initialize arrays
j_l			= np.zeros(n_spline)
j_l_spline	= np.zeros((n_hires, l_num))
Theta		= np.zeros((n_hires, l_num))
tck			= np.zeros((l_num, 3), dtype=object)
C_l			= np.zeros(l_num)

# Calculate spherical Bessel functions
for i in xrange(l_num):
	j_l[1:]	  = np.sqrt(np.pi / (2 * z_spline[1:])) * jv(ls[i]+0.5, z_spline[1:])
	tck[i,:]  = splrep(z_spline, j_l)
	
eta_diff	= get_eta(0) - get_eta(x_hires)
bessel_arg	= np.array([[k] * eta_diff for k in k_hires]).transpose()

dx = x_hires[1] - x_hires[0]
dk = k_hires[1] - k_hires[0]

for l in xrange(l_num):
	print l
	# Calculate spherical Bessel function at given order l
	j_l_spline = splev(bessel_arg, tck[l,:])

	# "Integrate" to find transfer function (summing for speed)
	theta_integrand = S_hires * j_l_spline * dx
	Theta[:,l] = np.sum(theta_integrand, axis=0)

	# "Integrate" to find C_l (summing for speed)
	cl_integrand = (c * k_hires / H_0)**(n_s - 1.0) * Theta[:,l] *\
					Theta[:,l] / k_hires * dk
	C_l[l] = np.sum(cl_integrand)
			
C_l *= ls * (ls + 1) / (2 * np.pi)

# New l-grid
ls2 = np.arange(2,1201)

# Spline Cl
tck_cl	= splrep(ls, C_l)
C_l2	= splev(ls2, tck_cl)

# Calculate normalization constant
norm = 5775.0 / np.max(C_l2)

end = time.time()

if __name__ == "__main__":

	print "Cl module time: %g seconds" % (end - start)

	# Write unnormalized Cl to file
	write2file("../data/milestone4/C_l_unnormalized_Ob0065-Om017-n093.txt",\
		"Power spectrum for the CMB, default model: l C_l", ls2, C_l2)

	# Write normalized Cl to file
	write2file("../data/milestone4/C_l_Ob0065-Om017-n093.txt",\
			"Power spectrum for the CMB, default model: l C_l", ls2, C_l2*norm)

	# Write transfer functions to file
	k_list = [5, 1000, 2000, 3000, 4000, 4999]
	for k in k_list:
		for l in xrange(l_num):
		write2file("../data/milestone4/transfer_func_k=%g.txt" % k,\
				"Transfer function as function of l, for six different k's: l Theta_l",\
				ls, Theta[k,:])
		write2file("../data/milestone4/spec_int_k=%g.txt" % k,\
				"Spectrum integrand as function of l, for six different k's: l Theta_l",\
				ls, Theta[k,:] * Theta[k,:] * H_0 / (c * k_hires[k]))

	# Write spectrum integrand to file
	for l in xrange(l_num):
		if ls[l] in [2, 50, 200, 500, 800, 1200]:
		output_integrand = Theta[:,l] * Theta[:,l] / k_hires
			write2file("../data/milestone4/transfer_func_l=%g.txt" % ls[l],\
					"Transfer function for one multipole value: k Theta_l",\
					k_hires * c/H_0, Theta[:,l])
			write2file("../data/milestone4/spec_int_l=%g.txt" % ls[l],\
					"Spectrum integrand for one multipole value: k Theta_l^2/k",\
					k_hires * c/H_0,\
					ls[l] * (ls[l] + 1) * Theta[:,l] * Theta[:,l] *\
					H_0/(c * k_hires))
