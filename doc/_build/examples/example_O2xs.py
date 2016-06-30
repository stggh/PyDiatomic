# -*- coding: utf-8 -*-
"""
  CSE  - solve the coupled-channel time-independent Schrödinger equation
         using recipe of B.R. Johnson J Chem Phys 69, 4678 (1977).

  Stephen.Gibson@anu.edu.au
  2016
"""

import numpy as np
import matplotlib.pylab as plt
import time

import cse

evcm = 8065.541   # conversion factor eV -> cm-1

d = 'potentials/'  # directory name

wavelength = np.arange(110, 174.1, 0.1)  # nm

# initialize CSE problem - any missing essential parameters are requested
# mu - reduced mass
# eni - initial state guess energy
# VTI - initial state(s)
# VTf - final coupled states
# coupf - homogeneous coupling
# dipolemoment - transition moments 

X = cse.Xs(mu='O2', VTi=[d+'X3S-1.dat'], eni=800,
                    VTf=[d+'B3S-1.dat', d+'3P1.dat', d+'E3S-1.dat',
                         d+'3PR1.dat'],
                    coupf=[40, 4000, 0, 0, 7000, 0],
                    dipolemoment=[1, 0, 0, 0.3])
                    #, transition_energy=wavelength)  # <--- alternative direct call

print("CSE: calculating cross section speeded by Python multiprocessing",
      " Pool.map")
print("     from {:.2f} to {:.2f} in {:.2f} nm steps ... ".
      format(wavelength[0], wavelength[-1], wavelength[1]-wavelength[0]))

t0 = time.time()
X.calculate_xs(transition_energy=wavelength)
print("CSE: ...  in {:.2g} seconds".format(time.time()-t0))

print('CSE: E(v"={:d}) = {:.2f} cm-1, {:.3g} eV'.format(X.gs.node_count(), 
                                                   X.gs.cm, X.gs.energy))
# np.savetxt("example_O2xs.dat", np.column_stack((wav, xs)))

# graphics ---------------------------------------
ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
ax1 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)

X.wavenumber /= 1.0e4
X.total = np.zeros_like(X.wavenumber)
for j in range(X.nopen):
   X.total[:] += X.xs[:, j]
   if X.us.pecfs[j][-7] == 'S':
       ax0.plot(X.xs[:, j], X.wavenumber, label=X.us.pecfs[j], color='b')
   else:
       ax0.plot(X.xs[:, j], X.wavenumber, label=X.us.pecfs[j], color='r',
                                                                 ls='--')

#ax0.plot(X.total, X.wavenumber, ls='-', color='gray', label='total', alpha=0.3)
ax0.legend(loc=0, frameon=False, fontsize=10)
ax0.set_ylabel("wavnumber ($10^4$cm$^{-1}$)")
ax0.set_xlabel("cross section (cm$^{2}$)")
ax0.axis(xmin=1.5e-17, xmax=-0.1e-17, ymin=4, ymax=10)
ax0.set_title("photodissociation cross section", fontsize=12)

for j, pec in enumerate(X.gs.pecfs):
   ax1.plot(X.gs.R, X.gs.VT[j, j]*evcm, color='k', label=pec)

for j, pec in enumerate(X.us.pecfs):
   if X.us.pecfs[j][-7] == 'S':
       ax1.plot(X.us.R, X.us.VT[j, j]*evcm, 'b', label=pec)
   else:
       ax1.plot(X.us.R, X.us.VT[j, j]*evcm, 'r--', label=pec)

ax1.annotate('$X{}^{3}\Sigma_{g}^{-}$', (0.6, 55000), color='k')
ax1.annotate('$B{}^{3}\Sigma_{u}^{-}$', (1.7, 55000), color='b')
ax1.annotate('$E{}^{3}\Sigma_{u}^{-}$', (0.9, 72000), color='b')
ax1.annotate('${}^{3}\Pi$', (1.34, 65000), color='r')


ax1.set_title("diabatic PECs", fontsize=12)
ax1.axis(xmin=0.5, xmax=2, ymin=40000+X.gs.cm, ymax=100000+X.gs.cm)
ax1.set_xlabel("R ($\AA$)")
#ax1.set_ylabel("V (eV)")
ax1.axes.get_yaxis().set_visible(False)

plt.suptitle("example_O2xs.py", fontsize=12)

plt.savefig("data/example_O2xs.png", dpi=100)
plt.show()
