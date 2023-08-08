import numpy as np
import cse
import matplotlib.pyplot as plt

O2X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
O2B = cse.Cse('O2', dirpath='potentials', VT=['B3S-1.dat', 'E3S-1.dat'],
              coup=[4000])
O2BX = cse.Transition(O2B, O2X, dipolemoment=[1, 0],
           transition_energy=np.arange(57550, 90000, 100))  # cm⁻¹

plt.plot(O2BX.wavenumber, O2BX.xs[:, 0])  # '0' is 'B3S-1.dat' channel
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.ylabel('Cross section (cm$^{2}$)')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-19, -19))
plt.title('O$_{2}$ $^{3}\Sigma_{u}^{-}$ Rydberg-valence interaction')

plt.savefig('figures/O2_RVxs.svg')
plt.show()
