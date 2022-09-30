import numpy as np
import cse
import scipy.constants as C
from scipy.integrate.quadrature import simps
import matplotlib.pyplot as plt

a0 = C.physical_constants["Bohr radius"][0]
CONST = 2*(np.pi*C.e*a0)**2*1.0e4/3/C.epsilon_0

expt = np.array([3.71e-6, 3.17e-5, 1.39e-4, 4.18e-4, 9.77e-4, 1.89e-3, 3.14e-3])

dE = np.array([32034.15590, 32464.71681, 32888.62393, 33307.19551, 33720.24892,
      34129.97896, 34535.15555, 34929.08111, 35319.59321, 35707.24213,
      36090.29405, 36470.64901, 36839.27576, 37196.99640, 37547.66353,
      37896.62079, 38239.20336, 38572.60107, 38898.81684, 39224.72975,
      39549.29221, 39859.79136, 40162.99435, 40460.64881, 40744.01827,
      41021.15289, 41284.82504, 41540.45698, 41784.05146])

for Bpot in ['B3S-1part.dat']: 

    S2 = cse.Xs('S2', VTi=['X3S-1part.dat'], eni=365,
 		      VTf=['{:s}'.format(Bpot)], dipolemoment=['DBXA.dati'])

    R = S2.gs.R

    S2.calculate_xs(transition_energy=dE-S2.gs.cm)

    osc = S2.xs

    omega = Bpot[5]
    print(osc)
    

    plt.plot(osc, 'o-', label=r'{:s}'.format(Bpot.strip('.dat')))

plt.plot(expt, '+k', ms=8, label='expt.')

plt.title(r"S$_2$ $B-X$ oscillator strength - Partridge")
plt.legend()
plt.xlabel(r'$v^{\prime}$')
plt.ylabel(r'$f_v$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.savefig("S2-osc.png", dpi=100)
plt.show()
