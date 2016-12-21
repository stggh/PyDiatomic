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

wavenumber = np.arange(57200, 85000, 100)

# initialize CSE problem - any missing essential parameters are requested
# mu - reduced mass
# eni - initial state guess energy
# VTI - initial state(s)
# VTf - final coupled states
# coupf - homogeneous coupling
# dipolemoment - transition moments 

O2 = cse.Xs(mu='O2', VTi=['potentials/X3S-1.dat'], eni=800,
                     VTf=['potentials/B3S-1.dat'], 
                     dipolemoment=['transitionmoments/dipole_b_valence.dat'])

print()
print(r"O2 Schumann-Runge continuum {:d} to {:d}, step {:d} (cm-1) ..."
      .format(wavenumber[0], wavenumber[-1], wavenumber[1]-wavenumber[0]))
      
tstart = time.time()
O2.calculate_xs(transition_energy=wavenumber)
tend = time.time()
print("    in {:.1f} seconds\n".format(tend-tstart))
print(" E(v\"=0) = {:8.2f} (cm-1)\n".format(O2.gs.cm))


wfX = np.transpose(O2.gs.wavefunction)[0][0]

R = O2.gs.R
v = []
fcf = []
Ev = []
# transition energies in cm-1
print("  v'   Ev'(cm-1)  Franck-Condon")
for e in [50145, 50832, 51497, 52139, 52756, 53347, 53910, 54443,
          54944, 55410, 55838, 56227, 56572, 56873, 57128, 57338,
          57507, 57640, 57743, 57820, 57874, 57908]:

    O2.us.solve(e)
    wfB = np.transpose(O2.us.wavefunction)[0][0]

    olap = (wfB * wfX)**2
    FCF = simps(olap, R)/10  

    v.append(O2.us.vib)
    Ev.append(O2.us.cm - O2.gs.cm)
    fcf.append(FCF)
    print(" {:2d}    {:8.2f}    {:5.2e}".format(v[-1], Ev[-1], fcf[-1]))


spl = InterpolatedUnivariateSpline(Ev, v, k=1)

dvdE = spl.derivative()(Ev)

plt.semilogy(Ev, fcf*dvdE/1.13e13, '+')
plt.semilogy(wavenumber, O2.xs)
plt.semilogy((57136.2, 57136.2), (1.0e-25, 2.0e-17), 'k--', lw=1)
plt.xlabel(r"wavenumber (cm$^{-1}$)")
plt.ylabel(r"cross section (cm$^{2}$)")
plt.title(r"O$_{2}$ $B{ }^{3}\Sigma_{u}^{-} - X{}^{3}\Sigma_{g}^{-}$")

plt.annotate(r"$f_{v^{\prime}0} \times \frac{dv^{\prime}}{dE}/1.13 \times"
             " 10^{13}$", (49000, 5.0e-18))
plt.annotate(r"$\sigma$", (70000, 1.0e-19))

plt.savefig("O2-Schumann-Runge.png", dpi=75)
plt.show()
