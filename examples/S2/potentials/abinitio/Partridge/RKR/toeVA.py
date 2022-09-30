import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

X = np.loadtxt("Partridge-X.csv", unpack=True)
B = np.loadtxt("Partridge-B.csv", unpack=True)
Bpp = np.loadtxt("Partridge-Bpp.csv", unpack=True)

Xrkr = np.loadtxt("X3S-1rkr.dat", unpack=True)
Brkr = np.loadtxt("B3S-1rkr.dat", unpack=True)

X[0] *= 0.529177
X[1] *= 27.2114
Xmin = X[1].min()
X[1] -= Xmin

B[0] *= 0.529177
B[1] *= 27.2114
B[1] -= Xmin

Bpp[0] *= 0.529177
Bpp[1] *= 27.2114
Bpp[1] -= Xmin

fig, ax = plt.subplots()

ax.plot(*X, '--', label=r'$X$part')
ax.plot(*B, '--', label=r'$B$part')
ax.plot(*Bpp, '--', label=r'$B^{\prime\prime}$part')
ax.plot(*Xrkr, 'C0', label=r'$Xrkr$')
ax.plot(*Brkr, 'C1', label=r'$Brkr$')
ax.legend()
ax.set_xlabel(r"internuclear distance ($\AA$)")
ax.set_ylabel(r"potential energy (eV)")
ax.axis(xmin=1.4, xmax=4, ymin=-0.1, ymax=8)

plt.show()
