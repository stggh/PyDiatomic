import numpy as np
import matplotlib.pyplot as plt

R, V = np.loadtxt("PX.csv", unpack=True)

R *= 0.52917721067
V *= 27.2114
V -= V.min()

RR, VR = np.loadtxt("../../X/X3S-1.dat", unpack=True)

plt.plot(R, V, label="Partridge Fig. 1 dig.")
plt.plot(RR, VR, label="RKR")

plt.legend(loc='best', frameon=False, labelspacing=0.1)

plt.xlabel(r"Internuclear distance ($\AA$)")
plt.ylabel(r"Potential energy (eV)")
plt.title(r"S$_2 X ^3\Sigma_g^-$")
plt.axis(xmin=1.2, xmax=4, ymin=-0.1, ymax=5)

plt.savefig("Partridge-RKR.png", dpi=75)
plt.show()
