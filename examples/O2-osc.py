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

dE = np.array([50145, 50832, 51497, 52139, 52756, 53347, 53910, 54443,
      54944, 55410, 55838, 56227, 56572, 56873, 57128, 57338,
      57507, 57640, 57743, 57820, 57874, 57908])

O2.calculate_xs(transition_energy=dE-O2.gs.cm)

a0 = C.physical_constants["Bohr radius"][0]
CONST = 2*(np.pi*C.e*a0)**2*1.0e4/3/C.epsilon_0


osc = O2.xs
print(osc)

fig, ax = plt.subplots(1, 2)

ax[0].plot(osc, 'bo')
ax[0].set_ylabel(r'$f_{v^\prime}$')
ax[0].set_xlabel(r'$v^\prime$')
ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

O2oscexpt = np.loadtxt("O2osc-expt.dat", unpack=True)
ax[0].errorbar(*O2oscexpt, fmt='o', color='C1', label='Yoshino')

ax[1].plot(osc, 'bo', label=r'Python')
ax[1].set_title(r"log scale")
ax[1].set_ylabel(r'$f_{v^\prime}$')
ax[1].set_xlabel(r'$v^\prime$')
ax[1].errorbar(*O2oscexpt, fmt='o', color='C1', label='Yoshino')
ax[1].set_yscale('log')
ax[1].legend()

plt.subplots_adjust(wspace=.4)
plt.suptitle(r"O$_2$ $B ^3\Sigma^-_u - X ^3\Sigma^-_g$ oscillator strengths")
plt.savefig("O2osc.png", dpi=100)
plt.show()
