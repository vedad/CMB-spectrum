#!/usr/bin/env python

"""
Created on 22 Jan. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt

inFile = np.loadtxt('../data/conformal_time.txt')
inFile_ipl = np.loadtxt('../data/conformal_time_ipl.txt')

# (empty) dictionary with data to be filled
data = {}

data['a_eta'] = inFile[:,0]
data['x_eta'] = inFile[:,1]
data['eta'] = inFile[:,2]
data['a_ipl'] = inFile_ipl[:,0]
data['x_ipl'] = inFile_ipl[:,1]
data['eta_ipl'] = inFile_ipl[:,2]

# Conformal time with a
# --------------------------
# Create figure
fig_a, ax_a = plt.subplots()

# Axis labels
ax_a.set_xlabel('$a$')
ax_a.set_ylabel('$\eta$ [Mpc]')

# Plotting
ax_a.plot(data['a_eta'], data['eta'], lw=1.5)

# Set to log-log scale
ax_a.set_xscale('log')
ax_a.set_yscale('log')

# Save figure and crop
fig_a.tight_layout()
fig_a.savefig('../results/conformal_time_a.pdf', bbox_inches='tight')
# --------------------------

# Conformal time with x
# --------------------------
fig_x, ax_x = plt.subplots()
ax_x.set_xlabel('$x$')
ax_x.set_ylabel('$\eta$ [Mpc]')
ax_x.plot(data['x_eta'], data['eta'], lw=1.5)
#plt.hold('on')
#ax_x.plot(data['x_ipl'], data['eta_ipl'], lw=1.5, label='Interpolated')

ax_x.set_xlim([-26,0])
ax_x.set_ylim([4e-6, 2e5])

ax_x.set_yscale('log')

#ax_x.legend(loc='best')

fig_x.tight_layout()
fig_x.savefig('../results/conformal_time_x.pdf', bbox_inches='tight')
# --------------------------
