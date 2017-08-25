import numpy as np
import cse
import matplotlib.pyplot as plt

######################################################
#
# Harmonic oscillator example
#
# Stephen.Gibson@anu.edu.au
#  24 Aug 2017
######################################################


def phi(n, R, alpha):
    y = R*np.sqrt(alpha)
    if n == 0:
        return (alpha/np.pi)**0.25 * np.exp(-y**2/2)
    if n == 1:
        return -(alpha/np.pi)**0.25 * np.sqrt(2) * y * np.exp(-y**2/2)
    if n > 1:
        return (alpha/np.pi)**0.25 * ((2*y**2-1)/np.sqrt(2)) * np.exp(-y**2/2)


R = np.linspace(-2, 2, 2001)
# quadratic PEC
V = R**2/10

X = cse.Cse(2, VT=[(R, V)])

for en in [0.01, 0.02, 0.04]:
    X.solve(en)
    print("E(v={:d}) = {:5.2f} cm-1".format(X.vib, X.cm))
    plt.plot(R, X.wavefunction[:, 0, 0]*400 + X.cm,
             label=r'$v={:d}$'.format(X.vib))
    plt.plot(R, phi(X.vib, R, 10)*400 + X.cm, 'k--')

plt.plot(np.NaN, np.NaN, 'k--', label=r'analytical')
plt.plot(R, V*8065.541)
plt.axis(xmin=-1.5, xmax=1.5, ymin=-800, ymax=1600)
plt.legend()
plt.title("Harmonic oscillator potential")
plt.ylabel(r"potential energy (cm$^{-1}$) / wavefunction $\times 400$")
plt.xlabel(r"$r$")
plt.show()
