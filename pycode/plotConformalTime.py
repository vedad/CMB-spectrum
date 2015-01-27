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

data = {}

data['a_eta'] = inFile[:,0]
data['eta'] = inFile[:,1]
data['a_ipl'] = inFile_ipl[:,0]
data['eta_ipl'] = inFile_ipl[:,1]

fig_eta, ax_eta = plt.subplots()
ax_eta.set_xlabel('$a$')
ax_eta.set_ylabel('$\eta$ [Mpc]')
ax_eta.plot(data['a_eta'], data['eta'], lw=1.5, label='Original')
plt.hold('on')
ax_eta.plot(data['a_ipl'], data['eta_ipl'], lw=1.5, label='Interpolated')

ax_eta.set_xscale('log')
ax_eta.set_yscale('log')

ax_eta.legend(loc='best')

fig_eta.tight_layout()
fig_eta.savefig('../results/conformal_time_compared.pdf')

print "Conformal time at a = 1 (today): %.2f Mpc" % data['eta'][-1]
