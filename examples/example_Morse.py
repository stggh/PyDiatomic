# -*- coding: utf-8 -*-
import numpy as np
import cse

import matplotlib.pyplot as plt

##################################################################
#
# example_Morse.py
#
# eigenvalues and eigenfunctions for a Morse potential energy curve
#
##################################################################


# internuclear distance in Angstroms
R = np.arange(0.5, 3.5, 0.005)

# simulates O2 X-state
VM = cse.analytical.Morse(r=R, re=1.21, De=5.21, Te=0.0, beta=2.65)

# PyDiatomic Cse class
morse = cse.Cse(mu='O2', VT=[(R, VM)])

plt.plot(R, VM, label="Morse PEC")

for en in [1000, 2000, 30000]:
    morse.solve(en)

    plt.plot(R, morse.wavefunction[:, 0, 0]/2+morse.energy,
             label=r'$v={:d}$'.format(morse.vib))


# plt.annotate(r"$E(v=${:d})$ = ${:.2f} eV, $B_{{v}} =$ {:.2f} cm$^{-1}$".
#              format(morse.vib, morse.energy, morse.Bv), (2, 1))

plt.axis(xmin=0.8, xmax=3.8, ymin=-2, ymax=8)
plt.ylabel("potential energy (eV)")
plt.xlabel(r"internuclear distance ($\AA$)")
plt.legend()

plt.savefig("output/example_Morse.png", dpi=75)
plt.show()
