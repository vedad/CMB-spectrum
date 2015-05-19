#!/usr/bin/env python

"""
Created on 13 May 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import splrep, splev, RectBivariateSpline
from scipy.special import jv, sph_jn
import time
from joblib import Parallel, delayed
import cPickle as pickle
from params import c, H_0, Omega_b, Omega_r, Omega_m, Omega_lambda, n_s
from CMBSpectrum import x_start_rec
from CMBSpectrum import get_eta, write2file
from evolution import S_hires, n_hires, k_hires, x_hires, k_min, k_max

start = time.time()

#k_min	  = 0.1 * H_0 / c
#k_max	  = 1e3 * H_0 / c
#n_hires = 5000
#x_hires = np.linspace(x_start_rec, 0, n_hires)
#k_hires = np.linspace(k_min, k_max, n_hires)

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
	j_l[1:] = np.sqrt(np.pi / (2 * z_spline[1:])) * jv(ls[i]+0.5, z_spline[1:])
	tck[i,:] = splrep(z_spline, j_l)
	
eta_diff = get_eta(0) - get_eta(x_hires)
bessel_arg = np.array([[k] * eta_diff for k in k_hires]).transpose()

dx = x_hires[1] - x_hires[0]
dk = k_hires[1] - k_hires[0]
for l in xrange(l_num):
	print l
	# Calculate spherical Bessel function at given order l
	j_l_spline = splev(bessel_arg, tck[l,:])

	# Integrate to find transfer function
	theta_integrand = S_hires * j_l_spline * dx
	if ls[l] == 100:
		write2file("../data/milestone4/sph_bess_test.txt",\
				"Testing integrand", x_hires, j_l_spline[:,1700])
	Theta[:,l] = np.sum(theta_integrand, axis=0)

	# Integrate to find C_l
	cl_integrand = (c * k_hires / H_0)**(n_s - 1.0) * Theta[:,l] *\
					Theta[:,l] / k_hires * dk

	print np.shape(cl_integrand)
	if ls[l] == 100:
		output_integrand = Theta[:,l] * Theta[:,l] / k_hires /\
				(1./H_0 * 1e-6 * c)
		write2file("../data/milestone4/cl_integrand.txt",\
				"Testing C_l integrand", k_hires * c / H_0,\
				output_integrand)
	C_l[l] = np.sum(cl_integrand)
			
C_l = C_l * ls * (ls + 1) / (2 * np.pi)

# New l-grid
ls2 = np.arange(2,1201)

# Spline Cls
tck_cl	= splrep(ls, C_l)
C_l2	= splev(ls2, tck_cl)

# Calculate normalization constant
norm = 5775.0 / np.max(C_l2)

write2file("../data/milestone4/C_l.txt",\
		"Power spectrum for the CMB: l C_l", ls2, norm * C_l2)

end = time.time()

print "Cl module time: %g seconds" % (end - start)


#n_spline  = 5400
#z_spline  = np.linspace(0,3500,n_spline)

# Initialize arrays
#j_l			= np.zeros((n_spline, l_num))
#j_l_spline	= np.zeros((n_hires, l_num))
#Theta		= np.zeros((n_hires, l_num))
#bessel_arg	= np.zeros((n_hires, n_hires))

# Initialize spherical Bessel functions

# Calculate spherical Bessel functions
#for i in xrange(l_num):
#	j_l[1:,i] = np.sqrt(np.pi / (2 * z_spline[1:])) * jv(ls[i]+0.5, z_spline[1:])

# Spline spherical Bessel functions
#get_j_l = RectBivariateSpline(z_spline, ls, j_l) # TODO: Use 1D spline. how?

#for k in xrange(n_hires):
#bessel_arg = k_hires * (get_eta(0) - get_eta(x_hires))

#print np.shape(bessel_arg)
#print np.where(bessel_arg == 0)
#print bessel_arg

#for l in xrange(l_num):
#	j_l_spline[:,k] = get_j_l(k_hires 
#for l in xrange(l_num):
#	j_l_spline[:-1,l] = get_j_l(bessel_arg[:-1,l], ls)

#print np.shape(j_l_spline)
#print j_l_spline
#integrand	= S_hires * j_l_spline * (x_hires[1] - x_hires[0])
#Theta		= np.sum(integrand, axis=0)

#if __name__ == "__main__":
	
	
#	j_l_spline_test = get_j_l(x_hires, ls)
#	print np.shape(j_l_spline_test)
#	print j_l_spline_test
#	for k in xrange(n_hires):
#		j_l_spline[:-1,k] = get_j_l(bessel_arg[:-1,k], ls[0])

#	for l in xrange(l_num):
#	bessel_arg = np.array([[eta_diff] * k_hires[k] for k in xrange(n_hires)])
#	bessel_arg = np.array([[k_hires[k] for k in xrange(n_hires)] * eta_diff])
#	bessel_arg = np.array([[k_hires[k]] * eta_diff for k in xrange(n_hires)])
#	bessel_arg = np.array([[get_eta(0) - get_eta(x_hires)] * k_hires[k] for k\
#			in xrange(n_hires)]).transpose()
#	bessel_arg = bessel_arg.transpose()
#	j_l_spline = get_j_l(bessel_arg
	

#	print np.shape(bessel_arg)
#	print bessel_arg
#	print np.shape(j_l_spline)
#	print j_l_spline

#	for l in xrange(l_num):
#		write2file("../data/milestone4/Theta/Theta_" + str(ls[l]) + ".txt",\
#				"Data for transfer function for given l as function of k: k Theta",\
#				k_hires, Theta[:,l])


"""
module cl_mod
  use healpix_types
  use evolution_mod
  use sphbess_mod
  implicit none

contains

  ! Driver routine for (finally!) computing the CMB power spectrum
  subroutine compute_cls
    implicit none

    integer(i4b) :: i, j, l, l_num, x_num, n_spline
    real(dp)     :: dx, S_func, j_func, z, eta, eta0, x0, x_min, x_max, d, e
    integer(i4b), allocatable, dimension(:)       :: ls
    real(dp),     allocatable, dimension(:)       :: integrand
    real(dp),     pointer,     dimension(:,:)     :: j_l, j_l2
    real(dp),     pointer,     dimension(:)       :: x_arg, int_arg, cls, cls2, ls_dp
    real(dp),     pointer,     dimension(:)       :: k, x
    real(dp),     pointer,     dimension(:,:,:,:) :: S_coeff
    real(dp),     pointer,     dimension(:,:)     :: S, S2
    real(dp),     allocatable, dimension(:,:)     :: Theta
    real(dp),     allocatable, dimension(:)       :: z_spline, j_l_spline, j_l_spline2
    real(dp),     allocatable, dimension(:)       :: x_hires, k_hires

    real(dp)           :: t1, t2, integral
    logical(lgt)       :: exist
    character(len=128) :: filename
    real(dp), allocatable, dimension(:) :: y, y2

    ! Set up which l's to compute
    l_num = 44
    allocate(ls(l_num))
    ls = (/ 2, 3, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, &
         & 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, &
         & 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200 /)

    ! Task: Get source function from evolution_mod


    ! Task: Initialize spherical Bessel functions for each l; use 5400 sampled points between 
    !       z = 0 and 3500. Each function must be properly splined
    ! Hint: It may be useful for speed to store the splined objects on disk in an unformatted
    !       Fortran (= binary) file, so that these only has to be computed once. Then, if your
    !       cache file exists, read from that; if not, generate the j_l's on the fly.
    n_spline = 5400
    allocate(z_spline(n_spline))    ! Note: z is *not* redshift, but simply the dummy argument of j_l(z)
    allocate(j_l(n_spline, l_num))
    allocate(j_l2(n_spline, l_num))


    ! Overall task: Compute the C_l's for each given l
    do l = 1, l_num

       ! Task: Compute the transfer function, Theta_l(k)


       ! Task: Integrate P(k) * (Theta_l^2 / k) over k to find un-normalized C_l's


       ! Task: Store C_l in an array. Optionally output to file

    end do


    ! Task: Spline C_l's found above, and output smooth C_l curve for each integer l


  end subroutine compute_cls
  
end module cl_mod
"""
