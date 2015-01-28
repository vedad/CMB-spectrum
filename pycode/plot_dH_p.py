#!/usr/bin/env python

"""
Created on 27 Jan. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt

inFile = np.loadtxt('../data/hubble_constant_deriv.txt')
#inFile2 = np.loadtxt('../data/hubble_constant_deriv_num.txt')

data = {}

data['a_eta'] = inFile[:,0]
data['dH'] = inFile[:,1]
#data['dH_num'] = inFile2[:,1]

fig_dH, ax_dH = plt.subplots()
ax_dH.set_xlabel('$a$')
ax_dH.set_ylabel('d$\mathcal{H}$/d$x$')
ax_dH.plot(data['a_eta'], data['dH'], lw=1.5)
#plt.hold('on')
#ax_dH.plot(data['a_eta'], data['dH_num'], lw=1.5, label='Numerical')

ax_dH.set_xscale('log')
#ax_dH.set_yscale('log')

fig_dH.tight_layout()
fig_dH.savefig('../results/hubble_constant_deriv.pdf')

