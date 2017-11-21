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
    X.solve(E0+B0*J*(J+1), J)
    print("    {:2d}     {:8.3f}".format(J, X.cm))
    EJ.append(X.cm)

EJ = np.array(EJ)
JJ = Jrange*(Jrange+1)
slope, intercept, r_value, p_value, std_err = stats.linregress(JJ, EJ)
mx = EJ.mean()
sx2 = ((EJ-mx)**2).sum()
sd_intercept = std_err * np.sqrt(1/len(EJ) + mx*mx/sx2)
sd_slope = std_err * np.sqrt(1/sx2)

print("Bv={:8.5f}, slope={:9.7f}+-{:8.7f}, intercept={:9.7f}+-{:8.7f}"
      .format(B0, slope, sd_slope, intercept, sd_intercept))


plt.plot(JJ, EJ, 'oC0')
plt.plot(JJ, intercept+slope*JJ, 'C1-')
plt.xlabel("$J(J+1)$")
plt.ylabel("energy (cm$^{-1}$)")
plt.annotate("{:8.3f} + {:8.3f}J(J+1)".format(intercept, slope),
             (400, EJ.mean()))
plt.axis(xmin=-1)
plt.show()
