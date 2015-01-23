#!/usr/bin/env python

"""
Created on 22 Jan. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt

inFile = np.loadtxt('../data/conformal_time.txt')

data = {}

data['x_eta'] = inFile[:,0]
data['eta'] = inFile[:,1]

fig_eta, ax_eta = plt.subplots()
ax_eta.set_xlabel('$a$')
ax_eta.set_ylabel('$\eta$ [Mpc]')
ax_eta.plot(data['x_eta'], data['eta'], lw=1.5)

ax_eta.set_xscale('log')
ax_eta.set_yscale('log')

fig_eta.tight_layout()
fig_eta.savefig('../results/conformal_time.pdf')
