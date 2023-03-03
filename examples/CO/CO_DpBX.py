import numpy as np
import cse
import matplotlib.pyplot as plt
import scipy.constants as C

X = cse.Cse('CO', dirpath='potentials', suffix='.dat', VT=['X1S0'], en=1080)
X0 = X.results[0][0]
B0 = X.results[0][1]

B = cse.Cse('CO', dirpath='potentials', suffix='.dat', VT=['B1S0'])
Dp = cse.Cse('CO', dirpath='potentials', suffix='.dat', VT=['Dp1S0tb'])
Rx = B.R[np.abs(B.VT[0, 0] - Dp.VT[0, 0]).argmin()]

DpB = cse.Cse('CO', dirpath='potentials', suffix='.dat', VT=['B1S0', 'Dp1S0tb'],
             coup=[2900*np.exp(-((B.R-Rx)/0.3)**2)])

DpBX = cse.Transition(DpB, X, dipolemoment=[1, 0])

wn = np.arange(1e7/98, 1e7/95, 1)
xst = np.zeros_like(wn)
T = 80

for Jdd in range(20):
    for Jd in (Jdd-1, Jdd, Jdd+1):
        eni = X0 + B0*Jdd*(Jdd+1)

        DpBX.calculate_xs(transition_energy=wn, eni=eni, roti=Jdd, rotf=Jd,
                          honl=True)

        Boltz = (2*Jdd+1)*np.exp(-(DpBX.gs.cm-X0)*C.h*C.c*100/C.k/T)
        xst += DpBX.xs.sum(axis=1)

plt.plot(1e7/wn, xst)
plt.show()
