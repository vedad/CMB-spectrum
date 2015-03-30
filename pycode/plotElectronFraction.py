#!/usr/bin/env python

"""
Created on 17 Mar. 2015

Author: Vedad Hodzic
E-mail: vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator

datafile = np.loadtxt("../data/milestone2/electron_fraction.txt")

data = {}

data['x']	= datafile[:,0]
data['a']	= np.exp(data['x'])
data['z']	= 1./data['a'] - 1
data['X_e']	= datafile[:,1]

#print data['x']
#print data['a']
#print data['z']
#print data['X_e']

### Plotting
fig_z, ax_z = plt.subplots()

ax_z.set_xlabel('$z$')
ax_z.set_ylabel('$X_\mathrm{e}$')

ax_z.plot(data['z'], data['X_e'], lw=1.5)
ax_z.invert_xaxis()

ax_z.set_xlim([1900, 0])
ax_z.set_ylim([1.5e-4, 1.5])
#ax_z.set_xscale('log')
ax_z.set_yscale('log')

#plt.minorticks_on()

#plt.xticks(np.arange(1800, 0, 200))
plt.xticks([1800, 1400, 1000, 600, 200],\
			['$1800$', '$1400$', '$1000$',\
			'$600$', '$200$'])
plt.yticks([1e-3, 1e-2, 1e-1, 1],\
			['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'])

#ax_z.tick_params(axis='x',which='minor')#,bottom='off')
xminorLocator   = MultipleLocator(100)
ax_z.xaxis.set_minor_locator(xminorLocator)



fig_z.tight_layout()
fig_z.savefig('../results/milestone2/electron_fraction_z.pdf',\
		bbox_inches='tight')
#plt.show()
