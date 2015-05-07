#!/usr/bin/env python

"""
Created on 15 Apr. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np											# Functions, mathematical operators
import matplotlib.pyplot as plt								# Plotting
from scipy.integrate import odeint							# ODE solver
from scipy.interpolate import splrep, splev					# Spline interpolation
import time													# Time module for runtime
from params import c, H_0, Omega_b, Omega_r, Omega_m		# Module for physical constants and parameters
from CMBSpectrum import n_t, x_start_rec, x_t				# Import variables
from CMBSpectrum import get_H_scaled, get_dH_scaled,\
						get_eta, write2file					# Import functions
from recombination import get_dtau, get_ddtau				# Import functions

def system_rhs(y, x, k):
	
	# The parameters we are solving for
	_delta			= y[0]
	_delta_b		= y[1]
	_v				= y[2]
	_v_b			= y[3]
	_Phi			= y[4]
	_Theta_0		= y[5]
	_Theta_1		= y[6]
	_Theta_2		= y[7]
	_Theta_3		= y[8]
	_Theta_4		= y[9]
	_Theta_5		= y[10]
	_Theta_6		= y[11]

	# Declare some frequently used parameters
	a	  = np.exp(x)
	H_p	  = get_H_scaled(x)
	dtau  = get_dtau(x)
	eta	  = get_eta(x)
	R	  = 4 * Omega_r / (3 * Omega_b * a)
	_Psi  = - _Phi - 12 * H_0 * H_0 * Omega_r * _Theta_2 /\
			(c * c * k * k * a * a)

	# The right-hand sides of the Einstein-Boltzmann equations
	Phi_rhs		  = _Psi - c * c * k * k * _Phi / (3 * H_p * H_p) +\
					H_0 * H_0 / (2 * H_p * H_p) * (Omega_m * _delta / a +\
					Omega_b * _delta_b / a + 4 * Omega_r * _Theta_0 / (a * a))
	Theta_0_rhs	  = - c * k * _Theta_1 / H_p - Phi_rhs
	Theta_1_rhs	  = c * k / (3.0 * H_p) * (_Theta_0 - 2 * _Theta_2 + _Psi) +\
					dtau * (_Theta_1 + _v_b / 3.0)
	Theta_2_rhs	  = c * k / (5.0 * H_p) * (2 * _Theta_1 - 3 * _Theta_3) +\
					0.9 * dtau * _Theta_2
	Theta_3_rhs	  = c * k / (7.0 * H_p) * (3 * _Theta_2 - 4 * _Theta_4) +\
					dtau * _Theta_3
	Theta_4_rhs	  = c * k / (9.0 * H_p) * (4 * _Theta_3 - 5 * _Theta_5) +\
					dtau * _Theta_4
	Theta_5_rhs	  = c * k / (11.0 * H_p) * (5 * _Theta_4 - 6 * _Theta_6) +\
					dtau * _Theta_5
	Theta_6_rhs	  = c * k * _Theta_5 / H_p - 7 * c * _Theta_6 / (H_p * eta) +\
					dtau * _Theta_6
	delta_rhs	  = c * k * _v / H_p - 3 * Phi_rhs
	v_rhs		  = - _v - c * k * _Psi / H_p
	delta_b_rhs	  = c * k * _v_b / H_p - 3 * Phi_rhs
	v_b_rhs		  = - _v_b - c * k * _Psi / H_p +\
					dtau * R * (3 * _Theta_1 + _v_b)

	return [delta_rhs, delta_b_rhs, v_rhs, v_b_rhs, Phi_rhs, Theta_0_rhs,\
			Theta_1_rhs, Theta_2_rhs, Theta_3_rhs, Theta_4_rhs, Theta_5_rhs,\
			Theta_6_rhs]#, _Psi]

def system_rhs_tc(y, x, k):

	# The parameters we are solving for
	_delta			= y[0]
	_delta_b		= y[1]
	_v				= y[2]
	_v_b			= y[3]
	_Phi			= y[4]
	_Theta_0		= y[5]
	_Theta_1		= y[6]

	# Declare some frequently used parameters
	a	  = np.exp(x)
	H_p	  = get_H_scaled(x)
	eta	  = get_eta(x)
	dtau  = get_dtau(x)
	ddtau = get_ddtau(x)
	dH_p  = get_dH_scaled(x)
	R	  = 4 * Omega_r / (3 * Omega_b * a)
	Theta_2	= - 20 * c * k * _Theta_1 / (45 * H_p * dtau)
	Theta_3 = - 3 * c * k * Theta_2 / (7 * H_p * dtau)
	Theta_4 = - 4 * c * k * Theta_3 / (9 * H_p * dtau)
	Theta_5 = - 5 * c * k * Theta_4 / (11 * H_p * dtau)
	Theta_6	= - 6 * c * k * Theta_5 / (13 * H_p * dtau)
	_Psi	= - _Phi - 12 * H_0 * H_0 * Omega_r * Theta_2 /\
				(c * c * k * k * a * a)
				
	# The right-hand sides of the Einstein-Boltzmann equations
	Phi_rhs		  = _Psi - c * c * k * k * _Phi / (3 * H_p * H_p) +\
					H_0 * H_0 / (2 * H_p * H_p) * (Omega_m * _delta / a +\
					Omega_b * _delta_b / a + 4 * Omega_r * _Theta_0 / (a * a))
	Theta_0_rhs	  = - c * k * _Theta_1 / H_p - Phi_rhs
	q			  = -(((1 - 2 * R) * dtau + (1 + R) * ddtau) *\
					(3 * _Theta_1 + _v_b) - c * k * _Psi / H_p +\
					(1 - dH_p/H_p) * c * k / H_p * (- _Theta_0 + 2 * Theta_2) -\
					c * k * Theta_0_rhs / H_p) /\
					((1 + R) * dtau + dH_p / H_p - 1)
	v_b_rhs		  = 1.0 / (1 + R) * (- _v_b - c * k * _Psi / H_p +\
					R * (q + c * k / H_p * (- _Theta_0 + 2 * Theta_2) -\
					c * k * _Psi / H_p))
	Theta_1_rhs	  = (q - v_b_rhs) / 3.0
	delta_rhs	  = c * k * _v / H_p - 3 * Phi_rhs
	v_rhs		  = - _v - c * k * _Psi / H_p
	delta_b_rhs	  = c * k * _v_b / H_p - 3 * Phi_rhs

	return [delta_rhs, delta_b_rhs, v_rhs, v_b_rhs, Phi_rhs, Theta_0_rhs,\
			Theta_1_rhs]

def get_tight_coupling_time(k):
	"""
	Computes the time at which tight coupling ends.
	-----------------------------------------------
	k		  : float
	returns	  :	float
	-----------------------------------------------
	"""
	# Start with x_init and add a dx instead.
	dx	  = 0.01
	x	  = x_init

	tight_coupling	= True

	while tight_coupling:
		if x >= x_start_rec:
			print "Third test."
			return x_start_rec
		elif abs(c * k / (get_H_scaled(x) * get_dtau(x))) > 0.1:
			print "First test."
			return x
		elif abs(get_dtau(x)) < 10:
			print "Second test."
			return x
		x += dx

start = time.time()

# Parameters
a_init	  = 1e-8
x_init	  =	np.log(a_init)
k_min	  = 0.1 * H_0 / c
k_max	  = 1e3 * H_0 / c
n_k		  = 101
lmax_int  =	6
ks		  = np.zeros(n_k)

# Values of k
for i in xrange(n_k):
	ks[i]  = k_min + (k_max - k_min)*(i/float(n_k-1))**2.

# Initialize arrays (names with _tc are for plotting tight coupling regime)
delta, delta_b, v, v_b, Phi,\
Psi, dPhi, dPsi, dv_b			= np.zeros((9,n_t,n_k))

delta_tc, delta_b_tc, v_tc,\
v_b_tc, Phi_tc, Psi_tc			= np.zeros((6,n_t,n_k))

Theta_tc						= np.zeros((n_t, lmax_int+1, n_k))
Theta, dTheta					= np.zeros((2, n_t, lmax_int+1, n_k))
x_tc							= np.linspace(x_init, x_start_rec, n_t)

# Initial conditions
Phi[0,:]	  = 1.
delta[0,:]	  = 1.5 * Phi[0,:]
delta_b[0,:]  = delta[0,:]
v[0,:]		  =	c * ks * Phi[0,:] / (2 * get_H_scaled(x_init))
v_b[0,:]	  = v[0,:]
Theta[0,0,:]  = 0.5 * Phi[0,:]
Theta[0,1,:]  = -c * ks * Phi[0,:] / (6 * get_H_scaled(x_init))
Theta[0,2,:]  =	-20 * c * ks * Theta[0,1,:] / (45 *\
				get_H_scaled(x_init) * get_dtau(x_init))
for l in xrange(3, lmax_int+1):
	Theta[0,l,:]  = -l/(2.0*l + 1) * c * ks * Theta[0,l-1,:] /\
					(get_H_scaled(x_init) * get_dtau(x_init))
Psi[0,:]	  = - Phi[0,:] - 12 * H_0 * H_0 * Omega_r * Theta[0,2,:] /\
				(c * c * ks * ks * a_init * a_init)


# For holding the solutions of ODE system
y_tight_coupling_0, y_tight_coupling  = np.zeros((2,7))
ymid0, ymid, y0, y					  = np.zeros((4,12))

for k in xrange(n_k):
	print k
	k_current = ks[k]

	### Tight coupling

	# Initial values
	y_tight_coupling0 = delta[0,k], delta_b[0,k], v[0,k], v_b[0,k],\
							Phi[0,k], Theta[0,0,k], Theta[0,1,k]

	# Find when tight coupling ends and the index
	x_tc_end = get_tight_coupling_time(k_current)
	tc_end = np.where(x_tc >= x_tc_end)[0][0]

	# Solve ODE system in tight coupling regime
	y_tight_coupling = odeint(system_rhs_tc, y_tight_coupling0, x_tc[:tc_end+1],\
						args=(k_current,))

	Y_tc = [y_tight_coupling[:,0], y_tight_coupling[:,1], y_tight_coupling[:,2],\
			y_tight_coupling[:,3], y_tight_coupling[:,4],\
			y_tight_coupling[:,5], y_tight_coupling[:,6]]
	
	# Store solutions
	delta_tc[:tc_end+1,k], delta_b_tc[:tc_end+1,k],\
	v_tc[:tc_end+1,k], v_b_tc[:tc_end+1,k],\
	Phi_tc[:tc_end+1,k], Theta_tc[:tc_end+1,0,k],\
	Theta_tc[:tc_end+1,1,k] = Y_tc
	
	for l in xrange(2,lmax_int+1):
		Theta_tc[:tc_end+1,l,k] = - l / (2.0*l + 1) * c * k_current *\
					Theta_tc[:tc_end+1,l-1,k] / (get_H_scaled(x_tc[:tc_end+1]) *\
							get_dtau(x_tc[:tc_end+1]))

	Psi_tc[:tc_end+1,k] = - Phi_tc[:tc_end+1,k] - 12 * H_0 * H_0 * Omega_r *\
							Theta_tc[:tc_end+1,2,k] /\
							(c * c * k_current * k_current *\
							np.exp(2*x_tc[:tc_end+1]))

	# Initial values for x_start_rec if last point in TC grid was x_start_rec
	y0 =	delta_tc[-1,k], delta_b_tc[-1,k], v_tc[-1,k], v_b_tc[-1,k],\
			Phi_tc[-1,k], Theta_tc[-1,0,k], Theta_tc[-1,1,k],\
			Theta_tc[-1,2,k], Theta_tc[-1,3,k], Theta_tc[-1,4,k],\
			Theta_tc[-1,5,k], Theta_tc[-1,6,k]

	### After tight coupling

	# Solve full system for the gap between x_tc_end until x_start_rec
	if x_tc_end is not x_start_rec:
		# Initial values for gap
		y_mid0	=	delta_tc[tc_end,k], delta_b_tc[tc_end,k],\
					v_tc[tc_end,k], v_b_tc[tc_end,k],\
					Phi_tc[tc_end,k], Theta_tc[tc_end,0,k],\
					Theta_tc[tc_end,1,k], Theta_tc[tc_end,2,k],\
					Theta_tc[tc_end,3,k], Theta_tc[tc_end,4,k],\
					Theta_tc[tc_end,5,k], Theta_tc[tc_end,6,k]


		# Solve full system in gap
		y_mid = odeint(system_rhs, y_mid0, x_tc[tc_end:], args=(k_current,))
			
		Y_mid = [y_mid[:,0], y_mid[:,1], y_mid[:,2],\
				y_mid[:,3], y_mid[:,4], y_mid[:,5],\
				y_mid[:,6], y_mid[:,7], y_mid[:,8],\
				y_mid[:,9], y_mid[:,10], y_mid[:,11]]

		# Store solutions in arrays
		delta_tc[tc_end:,k], delta_b_tc[tc_end:,k],\
		v_tc[tc_end:,k], v_b_tc[tc_end:,k],\
		Phi_tc[tc_end:,k], Theta_tc[tc_end:,0,k],\
		Theta_tc[tc_end:,1,k], Theta_tc[tc_end:,2,k],\
		Theta_tc[tc_end:,3,k], Theta_tc[tc_end:,4,k],\
		Theta_tc[tc_end:,5,k], Theta_tc[tc_end:,6,k] = Y_mid

		Psi_tc[tc_end:,k] =	- Phi_tc[tc_end:,k] - 12 * H_0 * H_0 * Omega_r *\
							Theta_tc[tc_end:,2,k] /\
							(c * c * k_current * k_current *\
							np.exp(2*x_tc[tc_end:]))

		# Initial values for x_start_rec
		y0 =	delta_tc[-1,k], delta_b_tc[-1,k], v_tc[-1,k], v_b_tc[-1,k],\
				Phi_tc[-1,k], Theta_tc[-1,0,k], Theta_tc[-1,1,k],\
				Theta_tc[-1,2,k], Theta_tc[-1,3,k], Theta_tc[-1,4,k],\
				Theta_tc[-1,5,k], Theta_tc[-1,6,k]
		# end if

	# Solve Einstein-Boltzmann equations from x_start_rec until today
	y = odeint(system_rhs, y0, x_t, args=(k_current,))

	Y = [y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6],\
			y[:,7], y[:,8], y[:,9], y[:,10], y[:,11]]

	# Store solutions in arrays
	delta[:,k], delta_b[:,k], v[:,k],\
	v_b[:,k], Phi[:,k], Theta[:,0,k],\
	Theta[:,1,k], Theta[:,2,k], Theta[:,3,k],\
	Theta[:,4,k], Theta[:,5,k], Theta[:,6,k]  = Y

	Psi[:,k] = - Phi[:,k] - 12 * H_0 * H_0 * Omega_r * Theta[:,2,k] /\
				(c * c * k_current * k_current * np.exp(2*x_t))

	# Store derivatives in arrays
	dv_b[:,k] = system_rhs(Y, x_t, k_current)[3]
	dPhi[:,k] = system_rhs(Y, x_t, k_current)[4]
	dTheta[:,0,k], dTheta[:,1,k],\
	dTheta[:,2,k], dTheta[:,3,k],\
	dTheta[:,4,k], dTheta[:,5,k],\
	dTheta[:,6,k] = system_rhs(Y, x_t, k_current)[5:12]
	dPsi[:,k] = - dPhi[:,k] - 12 * H_0 * H_0 * Omega_r * (dTheta[:,2,k] - 2*\
				Theta[:,2,k]) / (c * c * k_current * k_current * np.exp(2*x_t))

end = time.time()

if __name__ == "__main__":

	print "Runtime: %g seconds." % (end - start)
	start_write = time.time()
	
	# Merge tight coupling and post-tight coupling arrays
	delta_full		= np.concatenate((delta_tc, delta), axis=0)
	delta_b_full	= np.concatenate((delta_b_tc, delta_b), axis=0)
	v_full			= np.concatenate((v_tc, v), axis=0)
	v_b_full		= np.concatenate((v_b_tc, v_b), axis=0)
	Phi_full		= np.concatenate((Phi_tc, Phi), axis=0)
	Psi_full		= np.concatenate((Psi_tc, Psi), axis=0)
	Theta_full		= np.concatenate((Theta_tc, Theta), axis=0)
	Psi_full		= np.concatenate((Psi_tc, Psi), axis=0)
	x				= np.concatenate((x_tc, x_t), axis=0)

	for k in xrange(n_k):
		# Write solutions to files
		write2file("../data/milestone3/delta/delta_" + str(k) + ".txt",\
				"Data for CDM overdensity for one mode of k: x delta", x,
				delta_full[:,k])

		write2file("../data/milestone3/delta_b/delta_b_" + str(k) + ".txt",\
				"Data for baryon overdensity for one mode of k: x delta_b",\
				x, delta_b_full[:,k])

		write2file("../data/milestone3/v/v_" + str(k) + ".txt",\
				"Data for CDM velocity for one mode of k: x v",\
				x, v_full[:,k])

		write2file("../data/milestone3/v_b/v_b_" + str(k) + ".txt",\
				"Data for baryon velocity for one mode of k: x v_b",\
				x, v_b_full[:,k])
	
		write2file("../data/milestone3/Phi/Phi_" + str(k) + ".txt",\
				"Data for curvature potential for one mode of k: x Phi",\
				x, Phi_full[:,k])

		write2file("../data/milestone3/Psi/Psi_" + str(k) + ".txt",\
				"Data for gravitational potential for one mode of k: x Psi",\
				x, Psi_full[:,k])

		write2file("../data/milestone3/Theta0/Theta0_" + str(k) + ".txt",\
				"Data for Theta_0 for one mode of k: x Theta_0",\
				x, Theta_full[:,0,k])

		# Write derivatives to files
		write2file("../data/milestone3/dv_b/dv_b_" + str(k) + ".txt",\
				"Data for derivatives of baryons for one mode of k: x v_b",\
				x_t, dv_b[:,k])

		write2file("../data/milestone3/dPhi/dPhi_" + str(k) + ".txt",\
				"Data for derivative of Phi for one mode of k: x dPhi",\
				x_t, dPhi[:,k])

		write2file("../data/milestone3/dPsi/dPsi_" + str(k) + ".txt",\
				"Data for derivative of Psi for one mode of k: x dPsi",\
				x_t, dPsi[:,k])

		write2file("../data/milestone3/dTheta0/dTheta0_" + str(k) + ".txt",\
				"Data for derivative of monopole for one mode of k: x dTheta0",\
				x_t, dTheta[:,0,k])

	end_write = time.time()
	print "Writing time: %g seconds." % (end_write - start_write)
