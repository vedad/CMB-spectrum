#!/usr/bin/env python

"""
Created on 27 Jan. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt

inFile = np.loadtxt('../data/hubble_constant.txt')

# (empty) dictionary with data to be filled
data = {}

data['a'] = inFile[:,0]
data['z'] = 1./data['a']
data['x'] = inFile[:,1]
data['H'] = inFile[:,2]

# Figure for H(x)
# ---------------------
# Create figure and axes
fig_Hx, ax_Hx = plt.subplots(figsize=(8,6))

# Labels on axes
ax_Hx.set_xlabel('$x$')
ax_Hx.set_ylabel('$H$ [km s$^{-1}$ Mpc$^{-1}$]')

# Plot data
ax_Hx.plot(data['x'], data['H'], lw=1.5)

# log-scale on y axis
ax_Hx.set_yscale('log')

# Limits on axes
ax_Hx.set_xlim([-26, 0])

# Save figure and crop (remove whitespace)
fig_Hx.tight_layout()
fig_Hx.savefig('../results/hubble_constant_x.pdf', bbox_inches='tight')
# ---------------------

# Figure for H(a)
# ---------------------
fig_Ha, ax_Ha = plt.subplots()
ax_Ha.set_xlabel('$a$')
ax_Ha.set_ylabel('$H$ [km s$^{-1}$ Mpc$^{-1}$]')
ax_Ha.plot(data['a'], data['H'], lw=1.5)

ax_Ha.set_xscale('log')
ax_Ha.set_yscale('log')

fig_Ha.tight_layout()
fig_Ha.savefig('../results/hubble_constant_a.pdf', bbox_inches='tight')
# ---------------------


# Figure for H(z)
# ---------------------
fig_Hz, ax_Hz = plt.subplots(figsize=(8,6))
ax_Hz.set_xlabel('$1+z$')
ax_Hz.set_ylabel('$H$ [km s$^{-1}$ Mpc$^{-1}$]')
ax_Hz.plot(data['z'], data['H'], lw=1.5)

ax_Hz.set_xscale('log')
ax_Hz.set_yscale('log')

plt.setp(ax_Hz.get_yticklabels(), visible=False)

ax_Hz.set_xlim([-10, 3e10])

fig_Hz.tight_layout()
fig_Hz.savefig('../results/hubble_constant_z.pdf', bbox_inches='tight')
# ---------------------

# Figure for H(x) and H(z) as subplots
# ---------------------
fig_splt, ax_splt = plt.subplots(1,2,figsize=(16,6))
ax_splt[0].set_xlabel('$x$')
ax_splt[0].set_ylabel('$H$ [km s$^{-1}$ Mpc$^{-1}$]')
ax_splt[1].set_xlabel('$1+z$')
ax_splt[1].set_ylabel('$H$ [km s$^{-1}$ Mpc$^{-1}$]')

ax_splt[0].plot(data['x'], data['H'], lw=1.5)
ax_splt[1].plot(data['z'], data['H'], lw=1.5)

ax_splt[0].set_yscale('log')
ax_splt[1].set_xscale('log')
ax_splt[1].set_yscale('log')

ax_splt[0].set_xlim([-26,0])
ax_splt[1].set_xlim([-10, 3e10])

fig_splt.tight_layout()
fig_splt.savefig('../results/hubble_constant_splt.pdf', bbox_inches='tight')
# ---------------------
