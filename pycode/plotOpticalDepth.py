#!/usr/bin/env python

"""
Created on 19 Mar. 2015

Author: Vedad Hodzic
E-mail: vedad.hodzic@astro.uio.no
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogLocator

datafile = np.loadtxt("../data/milestone2/optical_depth.txt")

data = {}

data['x']	  = datafile[:,0]
data['a']	  = np.exp(data['x'])
data['z']	  = 1./data['a'] - 1
data['tau']	  = datafile[:,1]
data['dtau']  =	datafile[:,2]
data['ddtau'] = datafile[:,3]

### Plotting
fig_x, ax_x = plt.subplots()

ax_x.set_xlabel('$x$')
ax_x.set_ylabel('$\\tau$')

ax_x.plot(data['x'], data['tau'], lw=1.5, color='blue', label='$\\tau$')
ax_x.hold('on')
ax_x.plot(data['x'], abs(data['dtau']), lw=1.5, ls='dashed', color='red',\
		label="$|\\tau'|$")
ax_x.plot(data['x'], data['ddtau'], lw=1.5, ls='dotted', color='magenta',\
		label="$\\tau''$")

ax_x.set_xlim([-19, 1])
ax_x.set_ylim([1e-8, 1e8])
ax_x.set_yscale('log')

#plt.minorticks_on()
#plt.xticks(np.arange(-17.5, 0+1, 2.5))
#ax_x.xaxis.set_minor_locator(np.arange(-19, 1, 0.5))
#plt.yticks(np.arange(1e-6, 1e-6, 1e3)) 
plt.yticks([1e-6, 1e-3, 1, 1e3, 1e6],\
			['$10^{-6}$', '$10^{-3}$', '1', '$10^3$', '$10^6$'])

xmajorLocator   = MultipleLocator(2)
xmajorFormatter = FormatStrFormatter('%d')
xminorLocator   = MultipleLocator(0.5)
yminorLocator	= LogLocator()

ax_x.xaxis.set_major_locator(xmajorLocator)
ax_x.xaxis.set_major_formatter(xmajorFormatter)
ax_x.xaxis.set_minor_locator(xminorLocator)
#ax_x.yaxis.set_minor_locator(LogLocator(base=10.0, numticks=6))

ax_x.legend(bbox_to_anchor=(0.9,0.9))
fig_x.tight_layout()
fig_x.savefig('../results/milestone2/optical_depth_x.pdf',\
		bbox_inches='tight')
#plt.show()
