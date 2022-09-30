import numpy as np
import cse
import scipy.constants as C
from scipy.integrate.quadrature import simps
import matplotlib.pyplot as plt

a0 = C.physical_constants["Bohr radius"][0]
CONST = 2*(np.pi*C.e*a0)**2*1.0e4/3/C.epsilon_0

dE = np.array([31427.54, 31769.3, 32072.33, 32385.81, 32691.98, 32989.52,
               33278.72, 33561.81, 33834.60, 34095.87, 34310, 34588.05,
               34812.93, 35021.86, 35212.09, 35379.03, 35523.58,
               35649.88, 35737.14, 35847.11, 35938.72, 36040.43,
               36106.81, 36162.74, 36206.74])


for Bpot in ['Bpp_3P1_temp5.dat']:

    S2 = cse.Xs('S2', VTi=['X3S-1.dat'], eni=365,
 		      VTf=['{:s}'.format(Bpot)], dipolemoment=['DBppXA.dati'])
    R = S2.gs.R

    S2.calculate_xs(transition_energy=dE-S2.gs.cm)

    osc = S2.xs

    omega = Bpot[5]
    print(osc)
    

    plt.plot(osc, 'o-', label=r'{:s}'.format(Bpot.strip('.dat')))


plt.title(r"S$_2$ $B^{\prime\prime}-X$ oscillator strength - Partridge")
plt.legend()
plt.xlabel(r'$v^{\prime}$')
plt.ylabel(r'$f_v$')

plt.savefig("S2-oscPd.png", dpi=100)
plt.show()
