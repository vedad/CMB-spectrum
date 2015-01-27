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

data['a_eta'] = inFile[:,0]
data['H'] = inFile[:,1]

fig_H, ax_H = plt.subplots()
ax_H.set_xlabel('$a$')
ax_H.set_ylabel('$H$ [km s$^{-1}$ Mpc$^{-1}$]')
ax_H.plot(data['a_eta'], data['H'], lw=1.5)

ax_H.set_xscale('log')
ax_H.set_yscale('log')

fig_H.tight_layout()
fig_H.savefig('../results/hubble_constant.pdf')

"""
fig_Ha, ax_Ha = plt.subplots()
ax_Ha.set_xlabel('$a$')
ax_Ha.set_ylabel('$H$')
ax_Ha.set_title('Analytical solution')
ax_Ha.plot(ta, Ha
"""

print data['H'][-1]
