#!/usr/bin/env python

"""
Created on 15 Apr. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np							  # Functions, mathematical operators
import matplotlib.pyplot as plt				  # Plotting
from scipy.integrate import odeint			  # ODE solver
from scipy.interpolate import splrep, splev	  # Spline interpolation
import time									  # Time module for runtime
import params								  # Module for physical constants and parameters
from CMBSpectrum import n_t, get_H_scaled
from recombination import get_dtau

# Accuracy parameters
a_init	  = 1e-8
x_init	  =	np.log(a_init)
k_min	  = 0.1 * params.H_0 / params.c
k_max	  = 1e3 * params.H_0 / params.c
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
	v[0,i]		  =	params.c * ks[i] * Phi[0,i] / (2 * get_H_scaled(x_init))
	v_b[0,i]	  = v[0,i]
	Theta[0,0,i]  = 0.5 * Phi[0,i]
	Theta[0,1,i]  = -params.c * ks[i] * Phi[0,i] / (6 * get_H_scaled(x_init))
	Theta[0,2,i]  =	-20 * params.c * ks[i] * Theta[0,1,i] / (45 *\
					get_H_scaled(x_init) * get_dtau(x_init))
	for l in xrange(3, lmax_int+1):
		Theta[0,l,i]  = -l/(2*l + 1) * params.c * ks[i] * Theta[0,l-1,i] /\
						(get_H_scaled(x_init) * get_dtau(x_init))

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

for k in xrange(n_k):

	k_current = ks(k)
	h1		  =	1e-5

	y_tight_coupling[0]	= delta[0,k]
	y_tight_coupling[1]	= delta_b[0,k]
	y_tight_coupling[2]	= v[0,k]
	y_tight_coupling[3]	= v_b[0,k]
	y_tight_coupling[4]	= Phi[0,k]
	y_tight_coupling[5]	= Theta[0,0,k]
	y_tight_coupling[6]	= Theta[0,1,k]

	x_tc = get_tight_coupling_time(k_current)


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
