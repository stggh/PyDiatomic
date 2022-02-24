import numpy as np
import cse
from scipy import stats
import matplotlib.pyplot as plt

# evaluates E_J" vs J"(J"+1) for the O2 X-state

X = cse.Cse('O2', VT=['potentials/X3S-1.dat'])

X.solve(800)
E0 = X.cm
B0 = X.Bv


Jp = []
EJ = []
Jrange = np.arange(0, 30, 5)

print('    J"      E"(cm-1)')
for J in Jrange:
    en_guess = E0 + B0*J*(J+1)
    X.solve(en_guess, J)
    print(f'    {J:2d}     {X.cm:8.3f}')
    EJ.append(X.cm)

EJ = np.array(EJ)
JJ = Jrange*(Jrange+1)
slope, intercept, r_value, p_value, std_err = stats.linregress(JJ, EJ)
mx = EJ.mean()
sx2 = ((EJ-mx)**2).sum()
sd_intercept = std_err * np.sqrt(1/len(EJ) + mx*mx/sx2)
sd_slope = std_err * np.sqrt(1/sx2)

print(f'Bv={B0:5.3f}, slope={slope:5.3f}+-{sd_slope:5.3f}, '
      f'intercept={intercept:5.3f}+-{sd_intercept:5.3f}')

plt.plot(JJ, EJ, 'oC0')
plt.plot(JJ, intercept+slope*JJ, 'C1-')
plt.xlabel(r'$J(J+1)$')
plt.ylabel(r'energy (cm$^{-1}$)')
plt.annotate(f'{intercept:5.3f} + {slope:5.3f} J(J+1)', (400, EJ.mean()))
plt.axis(xmin=-1)

plt.tight_layout()
plt.savefig('output/example_O2X_rotation.svg')
plt.show()
