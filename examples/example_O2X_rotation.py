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
print('   J"     E"(cm-1)')
for J in Jrange:
     X.solve(E0+B0*J*(J+1), J)
     print("    {:d}   {:g}".format(J, X.cm))
     EJ.append(X.cm)

JJ = Jrange*(Jrange+1)
slope, intercept, r_value, p_value, std_err = stats.linregress(JJ, EJ)

print("B={:g}, slope={:g}, intercept={:g}".format(B0, slope, intercept))


plt.plot(JJ, EJ, 'ob')
plt.plot(JJ, intercept+slope*JJ, 'r-')
plt.xlabel("$J(J+1)$")
plt.ylabel("energy (cm$^{-1}$)")
plt.axis(xmin=-1)
plt.show()
