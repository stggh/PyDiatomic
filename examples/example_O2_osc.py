import numpy as np
import cse
import scipy.constants as C
from scipy.integrate.quadrature import simps
import matplotlib.pyplot as plt

# ground state
O2 = cse.Xs('O2', VTi=['potentials/X3S-1.dat'], eni=800,
                  VTf=['potentials/B3S-1.dat'],
                  dipolemoment=['transitionmoments/dipole_b_valence.dat'])
R = O2.gs.R

elevels = np.array([50145, 50832, 51497, 52139, 52756, 53347, 53910, 54443,
                    54944, 55410, 55838, 56227, 56572, 56873, 57128, 57338,
                    57507, 57640, 57743, 57820, 57874, 57908])

dE = elevels - O2.gs.cm
O2.calculate_xs(transition_energy = dE)
osc = O2.xs

Yv, Yosc, Yerr = np.loadtxt("data/O2osc-Yoshino.dat", unpack=True)
Yv0 = int(Yv[0])

print(r" v'     f_cse      f_Yoshino")
for v, f in enumerate(osc):
    if v in Yv:
        print("{:2d}   {:8.5e}   {}".format(v, f[0], Yosc[v-Yv0]))
    else:
        print("{:2d}   {:8.5e}   -".format(v, f[0]))

plt.plot(osc, 'o', label=r'PyDiatomic')
plt.errorbar(Yv, Yosc, Yerr, fmt='+', ms=5, label=r'Yoshino')

plt.title(r"O$_2$ $B\, ^3\Sigma_{u}^{-} - X\, ^3\Sigma_{g}^{-}$ $(v^\prime, v^{\prime\prime}=0)$")
plt.ylabel(r"oscillator strength")
plt.xlabel(r"$v^\prime$")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()
plt.yscale('log')

plt.savefig("output/example_O2_osc.png", dpi=100)
plt.show()
