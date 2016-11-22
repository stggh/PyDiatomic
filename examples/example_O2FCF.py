import numpy as np
import cse
import scipy.constants as C
from scipy.integrate.quadrature import simps
import matplotlib.pyplot as plt

# initial state 
O2X = cse.Cse('O2', VT=['potentials/X3S-1.dat'])
R = O2X.R

# final state
O2B = cse.Cse('O2', VT=['potentials/B3S-1.dat'])

O2X.solve(800)
wfX = np.transpose(O2X.wavefunction)[0][0]

print("  v'    FCF = |< v'| v\"=0 >|^2")

v = []; fcf = []
# transition energies in cm-1
for e in [50145, 50832, 51497, 52139, 52756, 53347, 53910, 54443,
          54944, 55410, 55838, 56227, 56572, 56873, 57128, 57338,
          57507, 57640, 57743, 57820, 57874, 57908]:

    O2B.solve(e)
    wfB = np.transpose(O2B.wavefunction)[0][0]

    olap = (wfB * wfX)**2
    FCF = simps(olap, R)/10  

    v.append(O2B.vib)
    fcf.append(FCF)
    print(" {:2d}    {:10.3e}".format(v[-1], FCF))

# fcf x1.18641 cf cse?

plt.plot(fcf, 'bo', label='PyDiatomic')
plt.title(r"O$_2$ FCF $B-X$")
plt.legend(frameon=False, numpoints=1)
plt.show()
