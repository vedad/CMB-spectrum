#!/usr/bin/env python

"""
Created on 28 Jan. 2015.

Author: Vedad Hodzic
E-mail:	vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt

inFile = np.loadtxt('../data/density_evolution.txt')

# (empty) dictionary with data to be filled
data = {}

data['a'] = inFile[:,0]
data['x'] = inFile[:,1]
data['Omega_m'] = inFile[:,2]
data['Omega_b'] = inFile[:,3]
data['Omega_r'] = inFile[:,4]
data['Omega_lambda'] = inFile[:,5]

# Density evolution with a
# --------------------------
# Create figure
fig_a, ax_a = plt.subplots()

# Axis labels
ax_a.set_xlabel('$a$')
ax_a.set_ylabel('Relative densities')

# Plotting
ax_a.plot(data['a'], data['Omega_m'], lw=1.5, label='$\Omega_\mathrm{m}$')
plt.hold('on')
ax_a.plot(data['a'], data['Omega_b'], lw=1.5, label='$\Omega_\mathrm{b}$')
ax_a.plot(data['a'], data['Omega_r'], lw=1.5, label='$\Omega_\mathrm{r}$')
ax_a.plot(data['a'], data['Omega_lambda'], lw=1.5, label='$\Omega_\Lambda$')

# x log-scale and set y-axis limits
ax_a.set_xscale('log')
ax_a.set_ylim([-0.05,1.05])

# Legend
ax_a.legend(loc='best')

# Save figure and crop
fig_a.tight_layout()
fig_a.savefig('../results/density_evolution_a.pdf', bbox_inches='tight')
# --------------------------

# Density evolution with x
# --------------------------
fig_x, ax_x = plt.subplots()
ax_x.set_xlabel('$x$')
ax_x.set_ylabel('Relative densities')
ax_x.plot(data['x'], data['Omega_m'], lw=1.5, label='$\Omega_\mathrm{m}$')
plt.hold('on')
ax_x.plot(data['x'], data['Omega_b'], lw=1.5, label='$\Omega_\mathrm{b}$')
ax_x.plot(data['x'], data['Omega_r'], lw=1.5, label='$\Omega_\mathrm{r}$')
ax_x.plot(data['x'], data['Omega_lambda'], lw=1.5, label='$\Omega_\Lambda$')

ax_x.set_xlim([-26,0])
ax_x.set_ylim([-0.05,1.05])

ax_x.legend(loc='best', bbox_to_anchor=(0.4,0.7))

fig_x.tight_layout()
fig_x.savefig('../results/density_evolution_x.pdf', bbox_inches='tight')
# --------------------------
