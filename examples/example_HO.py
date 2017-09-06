import numpy as np
import cse
import matplotlib.pyplot as plt
from scipy.special import hermite

######################################################
#
# Harmonic oscillator example
#
# Stephen.Gibson@anu.edu.au
#  24 Aug 2017
######################################################


def phi(v, R, alpha):
    y = R*np.sqrt(alpha)
    Nv = (alpha/np.pi)**0.25/np.sqrt(2**v*np.math.factorial(v))
    Hv = hermite(v)
    sum = 0.0
    for i, h in enumerate(Hv.coeffs[::-1]):
        sum += h*(y**i)
    return (-1)**v*sum*Nv*np.exp(-y**2/2)


R = np.linspace(-2, 2, 2001)
# quadratic PEC
V = R**2/10

X = cse.Cse(2, VT=[(R, V)])

fig, ax = plt.subplots()

print(" v    E(v) in cm-1")
for en in [0.01, 0.02, 0.04, 0.06, 0.08]:  # guess energies in eV
    X.solve(en)
    print("{:2d}  {:8.2f}".format(X.vib, X.cm))
    ax.plot(R, X.wavefunction[:, 0, 0]*400 + X.cm,
            label=r'$v={:d}$'.format(X.vib))
    ax.plot(R, phi(X.vib, R, 10)*400 + X.cm, 'k--')

ax.plot(np.NaN, np.NaN, 'k--', label=r'analytical')
ax.plot(R, V*8065.541)
ax.axis(xmin=-1.5, xmax=1.5, ymin=-800, ymax=1600)
ax.legend()
ax.set_title(r"Harmonic oscillator potential:"
             r" $\psi_v(x) = N_v H_v(x) e^{-x^2/2}$")
ax.set_ylabel(r"potential energy (cm$^{-1}$) / wavefunction $\times 400$")
ax.set_xlabel(r"$x$")

plt.savefig("output/example_HO.png", dpi=75)
plt.show()
