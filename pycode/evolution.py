#!/usr/bin/env python

"""
Created on 15 Apr. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np											# Functions, mathematical operators
import matplotlib.pyplot as plt								# Plotting
from numba import jit, float64
from scipy.integrate import odeint							# ODE solver
from scipy.interpolate import splrep, splev					# Spline interpolation
import time													# Time module for runtime
from params import c, H_0, Omega_b, Omega_r, Omega_m		# Module for physical constants and parameters
from CMBSpectrum import n_t, x_start_rec, x_t				# Import variables
from CMBSpectrum import get_H_scaled, get_dH_scaled, get_eta, write2file			# Import 
from recombination import get_dtau, get_ddtau

#x_init = np.log(1e-8)
#x_t		  = np.linspace(x_init, 0, n_t) # Grid for plotting through TC
#_a	  = np.exp(x_t)
#_H_p	  = get_H_scaled(x_t)
#_dtau  = get_dtau(x_t)
#_eta	  = get_eta(x_t)
#_dH_p  = get_dH_scaled(x_t)
#_ddtau = get_ddtau(x_t)

#@jit(nopython=True)
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
#	_Psi			= y[12]

	# Declare some frequently used parameters
#	a	  = _a[:t]c_end]
#	H_p	  = _H_p[:tc_end]
#	dtau  = _dtau[:tc_end]
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

#@jit
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
#	a	  = _a[:tc_end]
#	H_p	  = _H_p[:tc_end]
#	eta	  = _eta[:tc_end]
#	dtau  = _dtau[:tc_end]
#	ddtau = _ddtau[:tc_end]
#	dH_p  = _dH_p[:tc_end]
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

def get_Theta_2_tc(k, x, Theta_1):
	H_p	  = get_H_scaled(x)
	dtau  = get_dtau(x)

	return - 20 * c * k * Theta_1 / (45 * H_p * dtau)

def get_Theta_3_tc(k, x, Theta_2):
	H_p	  = get_H_scaled(x)
	dtau  = get_dtau(x)

	return - 3 * c * k * Theta_2 / (7 * H_p * dtau)

def get_Theta_4_tc(k, x, Theta_3):
	H_p	  = get_H_scaled(x)
	dtau  = get_dtau(x)

	return - 4 * c * k * Theta_3 / (9 * H_p * dtau)

def get_Theta_5_tc(k, x, Theta_4):
	H_p	  = get_H_scaled(x)
	dtau  = get_dtau(x)

	return - 5 * c * k * Theta_4 / (11 * H_p * dtau)

def get_Theta_6_tc(k, x, Theta_5):
	H_p	  = get_H_scaled(x)
	dtau  = get_dtau(x)

	return - 6 * c * k * Theta_5 / (13 * H_p * dtau)

def get_Psi(k, x, Theta_2, Phi):
	a = np.exp(x)

	return - Phi - 12 * H_0 * H_0 * Omega_r * Theta_2 /\
				(c * c * k * k * a * a)

def get_dPsi(k, x, dTheta_2, dPhi):
	a = np.exp(x)

	return - dPhi - 12 * H_0 * H_0 * Omega_r * dTheta_2 /\
				(c * c * k * k * a * a)


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
			return x-dx# - dx
		elif abs(get_dtau(x)) < 10:
			print "Second test."
			return x-dx# - dx
#			return x - dx
		x += dx

start = time.time()

### Parameters
a_init	  = 1e-8
x_init	  =	np.log(a_init)
k_min	  = 0.1 * H_0 / c
k_max	  = 1e3 * H_0 / c
n_k		  = 101
lmax_int  =	6
ks		  = np.zeros(n_k)
#x_tc	  = np.linspace(x_init, , n_t) # Grid for plotting through TC
#x_full	  = np.concatenate((x_tc,x_t), axis=0)

#a	  = np.exp(x_t)
#H_p	  = get_H_scaled(x_t)
#dtau  = get_dtau(x_t)
#eta	  = get_eta(x_t)
#dH_p  = get_dH_scaled(x_t)
#ddtau = get_ddtau(x_t)


### Values of k
for i in xrange(n_k):
	ks[i]  = k_min + (k_max - k_min)*(i/float(n_k-1))**2.

### Initialize arrays (try this new method instead of repeating stuff)
delta, delta_b, v, v_b, Phi,\
	Psi, dPhi, dPsi, dv_b		  = np.zeros((9,n_t,n_k))
delta_tc, delta_b_tc, v_tc,\
v_b_tc, Phi_tc, Psi_tc			  = np.zeros((6,n_t,n_k))
Theta_tc						  = np.zeros((n_t, lmax_int+1, n_k))
Theta, dTheta					  = np.zeros((2, n_t, lmax_int+1, n_k))
x_tc							  = np.linspace(x_init, x_start_rec, n_t)
#X								  = np.zeros((n_t, n_k))

#Theta	= np.zeros((n_t, lmax_int+1, n_k))
#delta	= np.zeros((n_t, n_k))
#delta_b	= np.zeros((n_t, n_k))
#v		= np.zeros((n_t, n_k))
#v_b		= np.zeros((n_t, n_k))
#Phi		= np.zeros((n_t, n_k))
#Psi		= np.zeros((n_t, n_k))
#dPhi	= np.zeros((n_t, n_k))
#dPsi	= np.zeros((n_t, n_k))
#dv_b	= np.zeros((n_t, n_k))
#dTheta	= np.zeros((n_t, lmax_int+1, n_k))

### Initial conditions
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


y_tight_coupling_0	= np.zeros(7)
y_tight_coupling	= np.zeros(7)
y0					= np.zeros(12)	
y					= np.zeros(12)

#print x_t
for k in xrange(n_k):
	print k
	k_current = ks[k]

	### Tight coupling

	# Initial values
	y_tight_coupling0 = delta[0,k], delta_b[0,k], v[0,k], v_b[0,k],\
							Phi[0,k], Theta[0,0,k], Theta[0,1,k]

	x_tc_end = get_tight_coupling_time(k_current)
#	x_tc_end = get_tight_coupling_time(k_current)
	print x_tc_end
	print x_start_rec
#	print x_tc
#	X[:,k]	  = np.linspace(x_init, x_tc_end, n_t)
#	x_tc_grid = np.linspace(x_init, x_tc, n_t)
#	x_full	  = np.concatenate((x_tc_grid, x_t), axis=0)
#	print x_tc
#	print X[:,k]
	tc_end = np.where(x_tc >= x_tc_end)[0][0]	# Index of TC end
#	tc_end += 1
	print tc_end

	# Solve Einstein-Boltzmann equations for tight coupling regime
#	y_tight_coupling = odeint(system_rhs_tc, y_tight_coupling0, X[:,k],\
#						args=(k_current,))

	y_tight_coupling = odeint(system_rhs_tc, y_tight_coupling0, x_tc[:tc_end+1],\
						args=(k_current,))

	Y_tc = [y_tight_coupling[:,0], y_tight_coupling[:,1], y_tight_coupling[:,2],\
			y_tight_coupling[:,3], y_tight_coupling[:,4],\
			y_tight_coupling[:,5], y_tight_coupling[:,6]]
	
	# Store solutions in respective arrays
#	delta[:tc_end,k], delta_b[:tc_end,k],\
#	v[:tc_end,k], v_b[:tc_end,k], Phi[:tc_end,k],\
#	Theta[:tc_end,0,k], Theta[:tc_end,1,k] = Y_tc

	delta_tc[:tc_end+1,k], delta_b_tc[:tc_end+1,k],\
			v_tc[:tc_end+1,k], v_b_tc[:tc_end+1,k], Phi_tc[:tc_end+1,k],\
			Theta_tc[:tc_end+1,0,k], Theta_tc[:tc_end+1,1,k] = Y_tc
	
#	delta_tc[:,k], delta_b_tc[:,k],\
#	v_tc[:,k], v_b_tc[:,k], Phi_tc[:,k],\
#	Theta_tc[:,0,k], Theta_tc[:,1,k] = Y_tc

	print Phi_tc[:,k]
	for l in xrange(2,lmax_int+1):
		Theta_tc[:tc_end+1,l,k] = - l / (2.0*l + 1) * c * k_current *\
					Theta_tc[:tc_end+1,l-1,k] / (get_H_scaled(x_tc[:tc_end+1]) *\
							get_dtau(x_tc[:tc_end+1]))

#	for l in xrange(2,lmax_int+1):
#		Theta_tc[:,l,k] = - l / (2.0*l + 1) * c * k_current *\
#							Theta_tc[:,l-1,k] / (get_H_scaled(X[:,k]) *\
#							get_dtau(X[:,k]))


#	Psi_tc[:,k]			  = get_Psi(k_current, X[:,k], Theta_tc[:,2,k],\
#							Phi_tc[:,k])
	Psi_tc[:tc_end+1,k]	  = get_Psi(k_current, x_tc[:tc_end+1],\
							Theta_tc[:tc_end+1,2,k], Phi_tc[:tc_end+1,k])
#	Psi_tc[:,k]			  = - Phi_tc[:,k] - 12 * H_0 * H_0 * Omega_r *\
#							Theta_tc[:,2,k] / (c * c * k_current * k_current *\
#							np.exp(X[:,k]) * np.exp(X[:,k]))

#	y0 = delta_tc[tc_end-1,k], delta_b_tc[tc_end,k],\
#			v_tc[tc_end-1,k], v_b_tc[tc_end-1,k],\
#			Phi_tc[tc_end-1,k], Theta_tc[tc_end-1,0,k],\
#			Theta_tc[tc_end-1,1,k], Theta_tc[tc_end-1,2,k],\
#			Theta_tc[tc_end-1,3,k], Theta_tc[tc_end-1,4,k],\
#			Theta_tc[tc_end-1,5,k], Theta_tc[tc_end-1,6,k]

	

	### After tight coupling

	# Initial values
	
	if x_tc_end < x_start_rec:
#		print X[:,k]
#	if x_tc_end is not x_start_rec:
#	if x_tc[tc_end,k] is not x_start_rec:
		y_mid0 = delta_tc[tc_end-1,k], delta_b_tc[tc_end-1,k],\
			v_tc[tc_end-1,k], v_b_tc[tc_end-1,k], Phi_tc[tc_end-1,k],\
			Theta_tc[tc_end-1,0,k], Theta_tc[tc_end-1,1,k], Theta_tc[tc_end-1,2,k],\
			Theta_tc[tc_end-1,3,k], Theta_tc[tc_end-1,4,k], Theta_tc[tc_end-1,5,k],\
			Theta_tc[tc_end-1,6,k]#, Psi_tc[-1,k]#, Theta
#		print y_mid0
#		y_mid0 = delta_tc[-1,k], delta_b_tc[-1,k],\
#			v_tc[-1,k], v_b_tc[-1,k],\
#			Phi_tc[-1,k], Theta_tc[-1,0,k],\
#			Theta_tc[-1,1,k], Theta_tc[-1,2,k],\
#			Theta_tc[-1,3,k], Theta_tc[-1,4,k],\
#			Theta_tc[-1,5,k], Theta_tc[-1,6,k]
#		print y_mid0

#		y_mid = odeint(system_rhs, y_mid0, [X[-1,k], x_t[0]],\
#				args=(k_current,))
		y_mid = odeint(system_rhs, y_mid0, x_tc[tc_end-1:], args=(k_current,))
			
		Y_mid = [y_mid[:,0], y_mid[:,1], y_mid[:,2],\
				y_mid[:,3], y_mid[:,4], y_mid[:,5],\
				y_mid[:,6], y_mid[:,7], y_mid[:,8],\
				y_mid[:,9], y_mid[:,10], y_mid[:,11]]
#		print Y_mid

#		y_test = [y_mid[:,0], y_mid[:,1], y_mid[:,2], y_mid[:,3], y_mid[:,4],\
#				y_mid[:,5], y_mid[:,6], y_mid[:,7], y_mid[:,8], y_mid[:,9],\
#				y_mid[:,10], y_mid[:,11]]
#		print y_test

#		y0 = [y_mid[-1,0], y_mid[-1,1], y_mid[-1,2], y_mid[-1,3], y_mid[-1,4],\
#			y_mid[-1,5], y_mid[-1,6], y_mid[-1,7], y_mid[-1,8], y_mid[-1,9],\
#			y_mid[-1,10], y_mid[-1,11]]
#		print y0
#		print y0
	
		delta_tc[tc_end-1:,k], delta_b_tc[tc_end-1:,k],\
				v_tc[tc_end-1:,k], v_b_tc[tc_end-1:,k],\
				Phi_tc[tc_end-1:,k], Theta_tc[tc_end-1:,0,k],\
				Theta_tc[tc_end-1:,1,k], Theta_tc[tc_end-1:,2,k],\
				Theta_tc[tc_end-1:,3,k], Theta_tc[tc_end-1:,4,k],\
				Theta_tc[tc_end-1:,5,k], Theta_tc[tc_end-1:,6,k] = Y_mid

		Psi_tc[tc_end-1:,k] = get_Psi(k_current, x_tc[tc_end-1:],\
								Theta_tc[tc_end-1:,2,k],\
								Phi_tc[tc_end-1:,k])

		y0 = delta_tc[-1,k], delta_b_tc[-1,k], v_tc[-1,k], v_b_tc[-1,k],\
				Phi_tc[-1,k], Theta_tc[-1,0,k], Theta_tc[-1,1,k],\
				Theta_tc[-1,2,k], Theta_tc[-1,3,k], Theta_tc[-1,4,k],\
				Theta_tc[-1,5,k], Theta_tc[-1,6,k]

	y0 = delta_tc[-1,k], delta_b_tc[-1,k], v_tc[-1,k], v_b_tc[-1,k],\
			Phi_tc[-1,k], Theta_tc[-1,0,k], Theta_tc[-1,1,k],\
			Theta_tc[-1,2,k], Theta_tc[-1,3,k], Theta_tc[-1,4,k],\
			Theta_tc[-1,5,k], Theta_tc[-1,6,k]
#	print "Phi[-1,k]:",Phi[-1,k]
#	print y0


#	y0 = delta_tc[-1,k], delta_b_tc[-1,k], v_tc[-1,k], v_b_tc[-1,k],\
#			Phi[-1,k], Theta_tc[-1,0,k], Theta_tc[-1,1,k],\
#			Theta_tc[-1,2,k], Theta_tc[-1,3,k], Theta_tc[-1,4,k],\
#			Theta_tc[-1,5,k], Theta_tc[-1,6,k]


#	delta[:tc_end,k], delta_b[:,k], v[:,k], v_b[:,k],\
#	Phi[:,k], Theta[:,0,k], Theta[:,1,k], Theta[:,2,k],\
#	Theta[:,3,k], Theta[:,4,k], Theta[:,5,k], Theta[:,6,k] = Y

	# Solve Einstein-Boltzmann equations from tight coupling until today
	y = odeint(system_rhs, y0, x_t, args=(k_current,))

	Y = [y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6],\
			y[:,7], y[:,8], y[:,9], y[:,10], y[:,11]]

#	delta[tc_end-1:,k], delta_b[tc_end-1:,k], v[tc_end-1:,k],\
#	v_b[tc_end-1:,k], Phi[tc_end-1:,k], Theta[tc_end-1:,0,k],\
#	Theta[tc_end-1:,1,k], Theta[tc_end-1:,2,k], Theta[tc_end-1:,3,k],\
#	Theta[tc_end-1:,4,k], Theta[tc_end-1:,5,k], Theta[tc_end-1:,6,k] = Y

#	Psi[tc_end-1:,k]	  = get_Psi(k_current, x_t[tc_end-1:],\
#							Theta[tc_end-1:,2,k],\
#							Phi[tc_end-1:,k])

	delta[:,k], delta_b[:,k], v[:,k], v_b[:,k],\
	Phi[:,k], Theta[:,0,k], Theta[:,1,k], Theta[:,2,k],\
	Theta[:,3,k], Theta[:,4,k], Theta[:,5,k], Theta[:,6,k] = Y

	Psi[:,k] = get_Psi(k_current, x_t, Theta[:,2,k], Phi[:,k])
#	print Phi[:,k]
				
#	Psi[:,k] = - Phi[:,k] - 12 * H_0 * H_0 * 

	# Store derivatives
#	dPhi[tc_end-1:,k]		  = system_rhs(Y, x_t[tc_end-1:],\
#								k_current)[4]
#	dv_b[tc_end-1:,k]		  = system_rhs(Y, x_t[tc_end-1:],\
#								k_current)[3]
#	dTheta[tc_end-1:,0,k], dTheta[tc_end-1:,1,k], dTheta[tc_end-1:,2,k],\
#	dTheta[tc_end-1:,3,k], dTheta[tc_end-1:,4,k], dTheta[tc_end-1:,5,k],\
#	dTheta[tc_end-1:,6,k] = system_rhs(Y, x_t[tc_end-1:], k_current)[5:12]
#	Psi[tc_end-1:,k]	  = y[:,12]

#	dPhi[:,k] = sy


end = time.time()

if __name__ == "__main__":

	print "Runtime: %g seconds." % (end - start)
	start_write = time.time()
	
	print delta_tc[:,0]
	print delta[:,0]
	delta_full		= np.concatenate((delta_tc, delta), axis=0)
	print delta_full[:,0]
	delta_b_full	= np.concatenate((delta_b_tc, delta_b), axis=0)
	v_full			= np.concatenate((v_tc, v), axis=0)
	v_b_full		= np.concatenate((v_b_tc, v_b), axis=0)
	Phi_full		= np.concatenate((Phi_tc, Phi), axis=0)
	Psi_full		= np.concatenate((Psi_tc, Psi), axis=0)
	Theta_full		= np.concatenate((Theta_tc, Theta), axis=0)
	Psi_full		= np.concatenate((Psi_tc, Psi), axis=0)
	x				= np.concatenate((x_tc, x_t), axis=0)
#	X_full = np.concatenate((X, x_t), axis=1) 

	for k in xrange(n_k):
#		x = np.concatenate((X[:,k], x_t), axis=0)

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

	end_write = time.time()
	print "Writing time: %g seconds." % (end_write - start_write)
