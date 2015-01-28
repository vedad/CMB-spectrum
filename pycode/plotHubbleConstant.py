#!/usr/bin/env python

"""
Created on 27 Jan. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt

inFile = np.loadtxt('../data/hubble_constant.txt')

data = {}

data['a'] = inFile[:,0]
data['z'] = 1./data['a']
data['x'] = inFile[:,1]
data['H'] = inFile[:,2]

# Figure for H(a)
# ---------------------
fig_Ha, ax_Ha = plt.subplots()
ax_Ha.set_xlabel('$a$')
ax_Ha.set_ylabel('$H$ [km s$^{-1}$ Mpc$^{-1}$]')
ax_Ha.plot(data['a'], data['H'], lw=1.5)

ax_Ha.set_xscale('log')
ax_Ha.set_yscale('log')

fig_Ha.tight_layout()
fig_Ha.savefig('../results/hubble_constant_a.pdf')
# ---------------------

# Figure for H(x)
# ---------------------
fig_Hx, ax_Hx = plt.subplots()
ax_Hx.set_xlabel('$x$')
ax_Hx.set_ylabel('$H$ [km s$^{-1}$ Mpc$^{-1}$]')
ax_Hx.plot(data['x'], data['H'], lw=1.5)

ax_Hx.set_yscale('log')

fig_Hx.tight_layout()
fig_Hx.savefig('../results/hubble_constant_x.pdf')
# ---------------------

# Figure for H(z)
# ---------------------
fig_Hz, ax_Hz = plt.subplots()
ax_Hz.set_xlabel('$1+z$')
ax_Hz.set_ylabel('$H$ [km s$^{-1}$ Mpc$^{-1}$]')
ax_Hz.plot(data['z'], data['H'], lw=1.5)

ax_Hz.set_xscale('log')
ax_Hz.set_yscale('log')

fig_Hz.tight_layout()
fig_Hz.savefig('../results/hubble_constant_z.pdf')
# ---------------------

print data['H'][-1]
