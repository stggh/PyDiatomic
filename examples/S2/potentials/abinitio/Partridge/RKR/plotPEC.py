import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

X = np.loadtxt("Partridge-X.csv", unpack=True)
B = np.loadtxt("Partridge-B.csv", unpack=True)
Bpp = np.loadtxt("Partridge-Bpp.csv", unpack=True)

Xrkr = np.loadtxt("X3S-1rkr.dat", unpack=True)
Brkr = np.loadtxt("B3S-1rkr.dat", unpack=True)

XS2 = np.loadtxt("../../../X/X3S-1.dat", unpack=True)

X[0] *= 0.529177
X[1] *= 27.2114
Xmin = X[1].min()
X[1] -= Xmin
np.savetxt("X3S-1part-dig-eV.dat", np.column_stack((X[0], X[1])))

B[0] *= 0.529177
B[1] *= 27.2114
B[1] -= Xmin
np.savetxt("B3S-1part-dig-eV.dat", np.column_stack((B[0], B[1])))

Bpp[0] *= 0.529177
Bpp[1] *= 27.2114
Bpp[1] -= Xmin
np.savetxt("Bpp3P0part-dig-eV.dat", np.column_stack((Bpp[0], Bpp[1])))

# interpolate
R = Xrkr[0]
spX = interp1d(*X, kind='cubic')
subr = np.logical_and(R > X[0][0], R < X[0][-1])
Xi = spX(R[subr])
np.savetxt("X3S-1parti.dat", np.column_stack((R[subr], Xi)))

spB = interp1d(*B, kind='cubic')
subr = np.logical_and(R > B[0][0], R < B[0][-1])
Bi = spB(R[subr])
np.savetxt("B3S-1parti.dat", np.column_stack((R[subr], Bi)))

# plots ------------
fig, ax = plt.subplots()

ax.plot(*X, '--', label=r'$X$part')
ax.plot(*B, '--', label=r'$B$part')
ax.plot(*Bpp, '--', label=r'$B^{\prime\prime}$part')
ax.plot(*Xrkr, 'C0', label=r'$Xrkr$')
ax.plot(*Brkr, 'C1', label=r'$Brkr$')
ax.plot(*XS2, 'C7:', ms=1, label=r'$Xcse$')
ax.legend()
ax.set_xlabel(r"internuclear distance ($\AA$)")
ax.set_ylabel(r"potential energy (eV)")
ax.axis(xmin=1.4, xmax=4, ymin=-0.1, ymax=8)

plt.savefig("PEC-Fig1.png", dpi=100)
plt.show()
