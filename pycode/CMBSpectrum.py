import numpy as np
import matplotlib.pyplot as plt
import params

class CMBSpectrum:

	def __init__(self, n_t, x_t, a_t, n_eta, x_eta, deta):
		"""
		An object (CMB spectrum) has attributes:
		n_t	  :	  Number of x-values
		x_t	  :	  Grid of x-values
		a_t	  :	  Grid of a-values (scale factor)

		n_eta :	  Number of grid points in conformal time
		x_eta :	  Grid points in conformal time
		eta	  :	  Conformal time at each grid point
		eta2  :	  Spline array of eta points (I think?)
		"""
		self.n_t, self.x_t, self.a_t, self.n_eta, self.x_eta, self.eta, self.eta2 \
				= n_t, x_t, a_t, n_eta, x_eta, eta, eta2
		
	def get_H(self, x):
		"""
		Computes the Hubble parameter H at given x.
		"""
		a = np.exp(x)
		return params.H_0 * np.sqrt((params.Omega_m + params.Omega_b) * a**(-3) + params.Omega_r *\
				a**(-4) + params.Omega_lambda)

	def get_H_scaled(self, x):
		"""
		Computes the scaled Hubble parameter H' = a*H at given x.
		"""
		a = np.exp(x)
		return a * get_H(x)

	def get_dH_scaled(self, x):

	def get_eta(self, x):
		"""
		Computes eta(x) using the previously computed spline function.
		"""
		
	
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
	x_pre = np.linspace(int(x_start_rec), int(x_end_rec), n1)
	x_post = np.linspace(int(x_end_rec), int(x_0), n2)
	x_t = np.concatenate((x_pre, x_post), axis=0)	# Concatenates two arrays

	# Grid for a
	a_t = np.exp(x_t) # Since x = ln(a)
	
	# Grid for x in conformal time
	x_eta = np.linspace(int(x_eta1), int(x_eta2), n_eta)

	eta = np.zeros(n_eta)
	eta[0] = 1./params.H_0 * np.sqrt(params.Omega_r)
	for i in xrange(n_eta):
		eta[i] = 





	
