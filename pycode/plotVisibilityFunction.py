#!/usr/bin/env python

"""
Created on 19 Mar. 2015

Author: Vedad Hodzic
E-mail: vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt

datafile = np.loadtxt("../data/milestone2/visibility_function.txt")

data = {}

data['x']	  = datafile[:,0]
data['a']	  = np.exp(data['x'])
data['z']	  = 1./data['a'] - 1
data['g']	  = datafile[:,1]
data['dg']	  =	datafile[:,2]
data['ddg']	  = datafile[:,3]

### Plotting
fig_x, ax_x = plt.subplots()

ax_x.set_xlabel('$x$')
ax_x.set_ylabel('$\\tilde{g}$')

ax_x.plot(data['x'], data['g'], lw=1.5, color='blue', label='$\\tilde{g}$')
ax_x.hold('on')
ax_x.plot(data['x'], data['dg']/10., lw=1.5, ls='dashed', color='red',\
		label="$\\tilde{g}'/10$")
ax_x.plot(data['x'], data['ddg']/300., lw=1.5, ls='dotted', color='magenta',\
		label="$\\tilde{g}''/300$")

ax_x.set_xlim([-7.55, -6])
ax_x.set_ylim([-4.0, 5.5])
#ax_x.set_yscale('log')

#plt.minorticks_on()
ax_x.tick_params(axis='x',which='minor')#,bottom='off')

plt.xticks(np.arange(-7.4, -5.8, 0.2))
#ax_x.xaxis.set_minor_locator(np.arange(-19, 1, 0.5))
plt.xticks([-7.4, -7.2, -7.0, -6.8, -6.6, -6.4, -6.2, -6.0],\
			['$-7.4$', '$-7.2$', '$-7$', '$-6.8$', '$-6.6$', '$-6.4$',\
			'$-6.2$', '$-6$'])

#plt.yticks(np.arange(-3.5, 5.5, 2)) 
plt.yticks([-2, 0, 2, 4],\
			[-2, 0, 2, 4])

ax_x.legend(bbox_to_anchor=(0.9,0.9))
fig_x.tight_layout()
fig_x.savefig('../results/milestone2/visibility_function.pdf',\
		bbox_inches='tight')
#plt.show()
