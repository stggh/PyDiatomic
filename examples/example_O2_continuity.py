# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
import time

import cse
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate.quadrature import simps

##########################################################################
# 
# O2 dissociation limit continuity as per 
#  Allison, Dalgarno, and Pasachoff  Planet Space Sci. 19, 1463-1473 (1971)
#  Fig. 3
#
# Stephen.Gibson@anu.edu.au
# 22 December 2016
##########################################################################


evcm = 8065.541   # conversion factor eV -> cm-1

continuum = np.arange(57200, 85000, 100)
bands = np.array([49357.4, 50044.9, 50710, 51351.5, 51968.4, 52559.6,
                  53122.6, 53655.3, 54156.5, 54622.1, 55051, 55439.5,
                  55784.8, 56085.6, 56340.7, 56551.1, 56720.1, 56852.7,
                  56955.2, 57032.5, 57086.9, 57120.7]) 

transition_energies = np.append(bands, continuum)

lb = len(bands)
v = np.arange(lb)

O2 = cse.Xs(mu='O2', VTi=['potentials/X3S-1.dat'], eni=800,
                     VTf=['potentials/B3S-1.dat'], 
                     dipolemoment=['transitionmoments/dipole_b_valence.dat'])

tstart = time.time()
O2.calculate_xs(transition_energy=transition_energies)
tend = time.time()
print("    in {:.1f} seconds\n".format(tend-tstart))
print(" E(v\"=0) = {:8.2f} (cm-1)\n".format(O2.gs.cm))

fcf = O2.xs[:lb, 0]

spl = InterpolatedUnivariateSpline(bands, v, k=1)
dvdE = spl.derivative()(bands)

plt.semilogy(bands, fcf * dvdE/1.13e12, '+')
plt.semilogy(continuum, O2.xs[lb:])
plt.semilogy((57136.2, 57136.2), (1.0e-25, 1.0e-18), 'k--', lw=1)
plt.xlabel(r"wavenumber (cm$^{-1}$)")
plt.ylabel(r"cross section (cm$^{2}$)")
plt.title(r"O$_{2}$ $B{ }^{3}\Sigma_{u}^{-} - X{}^{3}\Sigma_{g}^{-}$")

plt.annotate(r"$f_{v^{\prime}0} \frac{dv^{\prime}}{dE}/1.13 \times"
             " 10^{12}$", (49000, 3.0e-18), fontsize=12)
plt.annotate(r"$\sigma$", (70000, 5.0e-19), fontsize=12)

plt.savefig("data/example_O2_continuity.png", dpi=75)
plt.show()
