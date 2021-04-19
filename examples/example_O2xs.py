# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
import time

import cse

evcm = 8065.541   # conversion factor eV -> cm-1

wavelength = np.arange(110, 174.1, 0.1)  # nm

# O2 ground state X
X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)

# O2 upper coupled B-state
B = cse.Cse('O2', dirpath='potentials',
            VT=['B3S-1.dat', '3P1.dat', 'E3S-1.dat', '3PR1.dat'],
            coup=[40, 4000, 0, 0, 7000, 0])

# transition
BX = cse.Transition(B, X, dipolemoment=[1, 0, 0, 0.3])

print('CSE: calculating cross section speeded by Python multiprocessing'
      ' Pool.map')
print(f'     from {wavelength[0]:.0f} to {wavelength[-1]:.0f} in '
      f'{wavelength[1]-wavelength[0]:.2f} nm steps ... ')

t0 = time.time()
BX.calculate_xs(transition_energy=wavelength)
print(f'CSE: ...  in {time.time()-t0:.2g} seconds')

print(f'CSE: E(v"={BX.gs.vib:d}) = {BX.gs.cm:.2f} cm-1, {BX.gs.energy:.3g} eV')

# graphics ---------------------------------------
ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
ax1 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)

BX.wavenumber /= 1.0e4
BX.total = np.zeros_like(BX.wavenumber)
for j, xs in enumerate(BX.xs.T):
    if np.all(xs <= 0):
        continue
    BX.total[:] += xs
    if BX.us.pecfs[j][-7] == 'S':
        ax0.plot(BX.xs[:, j], BX.wavenumber, label=r'$^{3}\Sigma_{u}^{-}$',
                 color='C0')
    else:
        ax0.plot(BX.xs[:, j], BX.wavenumber, label=r'$^{3}\Pi$', color='C1',
                 ls='--')

ax0.legend(loc=0, frameon=False, fontsize=10)
ax0.set_ylabel("wavnumber ($10^4$cm$^{-1}$)")
ax0.set_xlabel("cross section (cm$^{2}$)")
ax0.axis(xmin=1.5e-17, xmax=-0.1e-17, ymin=4, ymax=10)
ax0.set_title("photodissociation cross section", fontsize=12)

for j, pec in enumerate(BX.gs.pecfs):
    ax1.plot(BX.gs.R, BX.gs.VT[j, j]*evcm, color='k', label=pec)

# adiabatic potential energy curves
# X.us.diabatic2adiabatic()
for j, pec in enumerate(BX.us.pecfs):
    if BX.us.pecfs[j][-7] == 'S':  # Sigma states
        ax1.plot(BX.us.R, BX.us.VT[j, j]*evcm, 'C0',
                 label=r'$^{3}\Sigma_{u}^{-}$')
        # ax1.plot(X.us.R, X.us.AT[j, j]*evcm, 'g', lw=2, label='adiabatic')
    else:
        ax1.plot(BX.us.R, BX.us.VT[j, j]*evcm, 'C1--', label=r'^{3}\Pi$')

ax1.annotate('$X{}^{3}\Sigma_{g}^{-}$', (0.6, 55000), color='k')
ax1.annotate('$B{}^{3}\Sigma_{u}^{-}$', (1.7, 55000), color='C0')
ax1.annotate('$E{}^{3}\Sigma_{u}^{-}$', (0.9, 72000), color='C0')
ax1.annotate('${}^{3}\Pi$', (1.34, 65000), color='C1')


ax1.set_title("diabatic PECs", fontsize=12)
ax1.axis(xmin=0.5, xmax=2, ymin=40000+BX.gs.cm, ymax=100000+BX.gs.cm)
ax1.set_xlabel("R ($\AA$)")
# ax1.set_ylabel("V (eV)")
ax1.axes.get_yaxis().set_visible(False)

plt.suptitle('example_O2xs.py', fontsize=12)

plt.savefig('output/example_O2xs.png', dpi=75)
plt.show()
