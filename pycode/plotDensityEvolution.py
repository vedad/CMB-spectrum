#!/usr/bin/env python

"""
Created on 28 Jan. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt

inFile = np.loadtxt('../data/density_evolution.txt')

data = {}

data['a'] = inFile[:,0]
data['Omega_m'] = inFile[:,1]
data['Omega_b'] = inFile[:,2]
data['Omega_r'] = inFile[:,3]
data['Omega_lambda'] = inFile[:,4]

fig_evo, ax_evo = plt.subplots()
ax_evo.set_xlabel('$a$')
ax_evo.set_ylabel('Relative densities')
ax_evo.plot(data['a'], data['Omega_m'], lw=1.5, label='$\Omega_\mathrm{m}$')
plt.hold('on')
ax_evo.plot(data['a'], data['Omega_b'], lw=1.5, label='$\Omega_\mathrm{b}$')
ax_evo.plot(data['a'], data['Omega_r'], lw=1.5, label='$\Omega_\mathrm{r}$')
ax_evo.plot(data['a'], data['Omega_lambda'], lw=1.5, label='$\Omega_\lambda$')

ax_evo.set_xscale('log')
ax_evo.set_ylim([-0.05,1.05])

ax_evo.legend(loc='best')

fig_evo.tight_layout()
fig_evo.savefig('../results/density_evolution.pdf')
