import numpy as np
import cse
import matplotlib.pyplot as plt

X = cse.Cse('CO', suffix='.dat', VT=['X1S0'], en=1080)
print(X)

B = cse.Cse('CO', suffix='.dat', VT=['B1S0'])
D = cse.Cse('CO', suffix='.dat', VT=['Dp1S0tb'])
Rx = B.R[np.abs(B.VT[0, 0] - D.VT[0, 0]).argmin()]
print(f'B-X crossing at {Rx:8.3f} Å')

BD = cse.Cse('CO', suffix='.dat', VT=['B1S0', 'Dp1S0tb'],
             coup=[2900*np.exp(-((B.R-Rx)/0.3)**2)])

BD.diabatic2adiabatic()

RR, DD = np.loadtxt('Dp1S0ck.dat',unpack=True)
evcm = X._evcm

for lbl in BD.statelabel:
    indx = BD.statelabel.index(lbl)
    plt.plot(BD.R, BD.VT[indx, indx]*evcm, f'C{indx}', label='diab.  '+lbl)
    plt.plot(BD.R, BD.AT[indx, indx]*evcm, f'--C{indx}', label='adiab. '+lbl)

plt.plot(RR, DD*evcm, 'C3:', label='$D^\prime$ labeled "ck"')
plt.axis(xmin=0.8, xmax=2.2, ymin=85000, ymax=115000)
plt.xlabel(r'Internuclear distance $R(\AA)$')
plt.ylabel(r'Potential energy (cm$^{-1}$)')
plt.legend()

plt.show()
