#!/usr/bin/env python

"""
Created on 15 Apr. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np											# Functions, mathematical operators
import matplotlib.pyplot as plt								# Plotting
import numba
from scipy.integrate import odeint							# ODE solver
from scipy.interpolate import splrep, splev					# Spline interpolation
import time													# Time module for runtime
from params import c, H_0, Omega_b, Omega_r, Omega_m		# Module for physical constants and parameters
from CMBSpectrum import n_t, x_start_rec					# Import variables
from CMBSpectrum import get_H_scaled, get_dH_scaled, get_eta, write2file			# Import 
from recombination import get_dtau, get_ddtau

def system_rhs(y, x, k):
	
	# The parameters we are solving for
	_Theta_0		= y[0]
	_Theta_1		= y[1]
	_Theta_2		= y[2]
	_Theta_3		= y[3]
	_Theta_4		= y[4]
	_Theta_5		= y[5]
	_Theta_6		= y[6]
	_delta			= y[5]
	_v				= y[6]
	_delta_b		= y[7]
	_v_b			= y[8]
	_Phi			= y[9]

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
	Theta_1_rhs	  = c * k / (3 * H_p) * (_Theta_0 - 2 * _Theta_2 + _Psi) +\
					dtau * (_Theta_1 + _v_b / 3.0)
	Theta_2_rhs	  = c * k / (5 * H_p) * (2 * _Theta_1 - 3 * _Theta3) +\
					0.9 * dtau * _Theta_2
	Theta_4_rhs	  = c * k / (7 * H_p) * (3 * _Theta_2 - 4 * _Theta_4) +\
					dtau * _Theta3
	Theta_4_rhs	  = c * k / (9 * H_p) * (4 * _Theta_3 - 5 * _Theta_5) +\
					dtau * _Theta_4
	Theta_5_rhs	  = c * k / (11 * H_p) * (5 * _Theta_4 - 6 * _Theta_6) +\
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
			Theta_6_rhs]

@numba.jit
def system_rhs_tc(y, x, k):

	# The parameters we are solving for
	_Theta_0		= y[0]
	_Theta_1		= y[1]
	_delta			= y[2]
	_v				= y[3]
	_delta_b		= y[4]
	_v_b			= y[5]
	_Phi			= y[6]

	# Declare some frequently used parameters
	a	  = np.exp(x)
	H_p	  = get_H_scaled(x)
	dH_p  = get_dH_scaled(x)
	dtau  = get_dtau(x)
	ddtau = get_ddtau(x)
	eta	  = get_eta(x)
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
	v_b_rhs		  = 1 / (1 + R) * (- _v_b - c * k * _Psi / H_p +\
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
		if abs(c * k / (get_H_scaled(x) * get_dtau(x))) > 0.1:
			print "First test."
			return x
		elif abs(get_dtau(x)) < 10:
			print "Second test."
			return x
		elif x >= x_start_rec:
			print "Third test."
			return x
		x += dx

start = time.time()

# Accuracy parameters
a_init	  = 1e-8
x_init	  =	np.log(a_init)
k_min	  = 0.1 * H_0 / c
k_max	  = 1e3 * H_0 / c
n_k		  = 100
lmax_int  =	6
ks		  = np.zeros(n_k)

for i in xrange(n_k):
	ks[i]  = k_min + (k_max - k_min)*(i/float(n_k))**2.

Theta	= np.zeros((n_t+1, lmax_int+1, n_k))
delta	= np.zeros((n_t+1, n_k))
delta_b	= np.zeros((n_t+1, n_k))
v		= np.zeros((n_t+1, n_k))
v_b		= np.zeros((n_t+1, n_k))
Phi		= np.zeros((n_t+1, n_k))
Psi		= np.zeros((n_t+1, n_k))
dPhi	= np.zeros((n_t+1, n_k))
dPsi	= np.zeros((n_t+1, n_k))
dv_b	= np.zeros((n_t+1, n_k))
dTheta	= np.zeros((n_t+1, lmax_int+1, n_k))

Phi[0,:]	  = 1.
delta[0,:]	  = 1.5 * Phi[0,:]
delta_b[0,:]  = delta[0,:]

for i in xrange(n_k):
	v[0,i]		  =	c * ks[i] * Phi[0,i] / (2 * get_H_scaled(x_init))
	v_b[0,i]	  = v[0,i]
	Theta[0,0,i]  = 0.5 * Phi[0,i]
	Theta[0,1,i]  = -c * ks[i] * Phi[0,i] / (6 * get_H_scaled(x_init))
	Theta[0,2,i]  =	-20 * c * ks[i] * Theta[0,1,i] / (45 *\
					get_H_scaled(x_init) * get_dtau(x_init))
	for l in xrange(3, lmax_int+1):
		Theta[0,l,i]  = -l/(2*l + 1) * c * ks[i] * Theta[0,l-1,i] /\
						(get_H_scaled(x_init) * get_dtau(x_init))

#print get_tight_coupling_time(ks[-1])
### Ask HK if values make sense
"""
print "v[0,:]: ", v[0,:]
print "v_b[0,:] ", v_b[0,:]
print "Theta[0,0,:] ", Theta[0,0,:]
print "Theta[0,1,:] ", Theta[0,1,:]
print "Theta[0,2,:] ", Theta[0,2,:]
print "Theta[0,3,:] ", Theta[0,3,:]
print "Theta[0,4,:] ", Theta[0,4,:]
print "Theta[0,5,:] ", Theta[0,5,:]
print "Theta[0,6,:] ", Theta[0,6,:]
"""

hmin			  = 0.0
y_tight_coupling  = np.zeros(7)
x_grid			  = np.linspace(x_init, 0, n_t+1)

for k in xrange(n_k):

	k_current = ks[k]
	h1		  =	1e-5

	y_tight_coupling[0]	= delta[0,k]
	y_tight_coupling[1]	= delta_b[0,k]
	y_tight_coupling[2]	= v[0,k]
	y_tight_coupling[3]	= v_b[0,k]
	y_tight_coupling[4]	= Phi[0,k]
	y_tight_coupling[5]	= Theta[0,0,k]
	y_tight_coupling[6]	= Theta[0,1,k]

	x_tc = get_tight_coupling_time(k_current)
	tc_end = np.where(x_grid > x_tc)[0][0]	# Index of TC end

	system_soln = odeint(system_rhs_tc, y_tight_coupling, x_grid[:tc_end],\
			args=(k_current,))
#	delta[:,k] = system_soln[
	delta[:tc_end, k]	  = system_soln[:,0]
	delta_b[:tc_end, k]	  = system_soln[:,1]
	v[:tc_end, k]		  = system_soln[:,2]
	v_b[:tc_end, k]		  = system_soln[:,3]
	Phi[:tc_end, k]		  = system_soln[:,4]
	Theta[:tc_end, 0, k]  = system_soln[:,5]
	Theta[:tc_end, 1, k]  = system_soln[:,6]

	print k

end = time.time()

if __name__ == "__main__":

	print "Runtime: %g seconds." % (end - start)

	for k in xrange(len(ks)):
		write2file("../results/milestone3/delta_" + str(k) + ".txt",\
				"Data for CDM density for one mode of k: x delta", x_grid,
				delta[:,k])

		write2file("../results/milestone3/Phi_" + str(k) + ".tx",\
				"Data for gravitational curvature potential for one mode of k:\
				 x Phi", x_grid, Phi[:,k])

"""
  subroutine integrate_perturbation_eqns
    implicit none

    integer(i4b) :: i, j, k, l
    real(dp)     :: x1, x2, x_init
    real(dp)     :: eps, hmin, h1, x_tc, H_p, dt, t1, t2

    real(dp), allocatable, dimension(:) :: y, y_tight_coupling, dydx

    x_init = log(a_init)
    eps    = 1.d-8
    hmin   = 0.d0

    allocate(y(npar))
    allocate(dydx(npar))
    allocate(y_tight_coupling(7))

    ! Propagate each k-mode independently
    do k = 1, n_k

       k_current = ks(k)  ! Store k_current as a global module variable
       h1        = 1.d-5

       ! Initialize equation set for tight coupling
       y_tight_coupling(1) = delta(0,k)
       y_tight_coupling(2) = delta_b(0,k)
       y_tight_coupling(3) = v(0,k)
       y_tight_coupling(4) = v_b(0,k)
       y_tight_coupling(5) = Phi(0,k)
       y_tight_coupling(6) = Theta(0,0,k)
       y_tight_coupling(7) = Theta(0,1,k)
       
       ! Find the time to which tight coupling is assumed, 
       ! and integrate equations to that time
       x_tc = get_tight_coupling_time(k_current)

       ! Task: Integrate from x_init until the end of tight coupling, using
       !       the tight coupling equations


       ! Task: Set up variables for integration from the end of tight coupling 
       ! until today
       y(1:7) = 
       y(8)   = 
       do l = 3, lmax_int
          y(6+l) = 
       end do

       
       do i = 1, n_t
          ! Task: Integrate equations from tight coupling to today

          ! Task: Store variables at time step i in global variables
          delta(i,k)   = 
          delta_b(i,k) = 
          v(i,k)       = 
          v_b(i,k)     = 
          Phi(i,k)     = 
          do l = 0, lmax_int
             Theta(i,l,k) = 
          end do
          Psi(i,k)     = 

          ! Task: Store derivatives that are required for C_l estimation
          dPhi(i,k)     = 
          dv_b(i,k)     = 
          dTheta(i,:,k) = 
          dPsi(i,k)     = 
       end do

    end do

    deallocate(y_tight_coupling)
    deallocate(y)
    deallocate(dydx)

  end subroutine integrate_perturbation_eqns
"""
