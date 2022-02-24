import numpy as np
import cse
import scipy.constants as C
from scipy.integrate import simps
import matplotlib.pyplot as plt

# ground state
O2X = cse.Cse('O2', VT=['potentials/X3S-1.dat'], en=800)
O2B = cse.Cse('O2', VT=['potentials/B3S-1.dat'])

O2 = cse.Transition(O2B, O2X, 
                    dipolemoment=['transitionmoments/dipole_b_valence.dat'])
R = O2.gs.R

# B-X transition energy guesses
bands = [49357, 50045, 50710, 51352, 51968, 52560, 53123, 53655,
         54157, 54622, 55051, 55439, 55785, 56086, 56341, 56551,
         56720, 56853, 56955, 57032, 57087, 57121]

O2.calculate_xs(transition_energy=bands)
osc = O2.xs

Yv, Yosc, Yerr = np.loadtxt("data/O2osc-Yoshino.dat", unpack=True)
Yv0 = int(Yv[0])

print(r" v'     f_cse      f_Yoshino")
for v, f in enumerate(osc):
    if v in Yv:
        print(f'{v:2d}   {f[0]:8.5e}   {Yosc[v-Yv0]}')
    else:
        print(f'{v:2d}   {f[0]:8.5e}   -')

plt.plot(osc, 'o', label=r'PyDiatomic')
plt.errorbar(Yv, Yosc, Yerr, fmt='+', ms=5, label=r'Yoshino')

plt.title(r'O$_2$ $B\, ^3\Sigma_{u}^{-} - X\, ^3\Sigma_{g}^{-}$ $(v^\prime,'
          r' v^{\prime\prime}=0)$')
plt.ylabel(r'oscillator strength')
plt.xlabel(r'$v^\prime$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.savefig('output/example_O2_osc.svg')
plt.show()
