import numpy as np
import matplotlib.pyplot as plt

Quick = np.loadtxt('Aexpt.dat', unpack=True)
Glenn = np.loadtxt('A-Glenn.dat')
Part = np.loadtxt('A-Part.dat')
vib, MatO0, eMatO0, MatO1, eMatO1 = np.loadtxt('A-Matsumi.dat', unpack=True)

plt.plot(Glenn, 'o', mfc='w', label='Glenn')
plt.plot(Part, '+', label='Partrigde calc.')
plt.errorbar(Quick[0], Quick[1], yerr=Quick[2], fmt='x', label='Quick')
plt.errorbar(vib, MatO0, yerr=eMatO0, fmt='^', label=r'Matsumi $\Omega=0$')
plt.errorbar(vib, MatO1, yerr=eMatO1, fmt='v', label=r'Matsumi $\Omega=1$')
plt.legend(fontsize='smaller', labelspacing=0.1)
plt.xlabel(r'$^\prime$')
plt.ylabel('life time (ns)')

plt.show()
