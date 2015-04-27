#!/usr/bin/env python

"""
Created on 22 Apr. 2015

Author: Vedad Hodzic
E-mail: vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator

data = {}

for i in xrange(100):
	datafile = np.loadtxt("../data/milestone3/Phi/Phi_" + str(i) + ".txt")
	data['k=' + str(i)] = {}
	data['k=' + str(i)]['x'] = datafile[:,0]
	data['k=' + str(i)]['Phi'] = datafile[:,1]

### Plotting
fig_Phi, ax_Phi = plt.subplots()

ax_Phi.set_xlabel('$x$')
ax_Phi.set_ylabel('$\Phi$')

plt.hold('on')
k_vals = [5, 15, 40, 60, 85, 95]
for k in k_vals:
	ax_Phi.plot(data['k=' + str(k)]['x'], data['k=' + str(k)]['Phi'], lw=1.5,\
			label=('$k=$ ' + str(k)))
#ax_Phi.plot(data['k=40'][:,0][:300], data['k=40'][:,1][:300], lw=1.5,
#label='$k_{40}$')

ax_Phi.legend(loc='best')
fig_Phi.tight_layout()
fig_Phi.savefig('../results/milestone3/Phi.pdf', bbox_inches='tight')

