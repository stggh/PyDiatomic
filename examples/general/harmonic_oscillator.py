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

print(' v       E     E-(v + ½)ωₑ in cm⁻¹')
for en in [220, 300, 400, 500, 700]:  # guess energies in cm⁻¹
    X.solve(en)
    en, v, wf = X.cm, X.vib, X.wavefunction[:, 0, 0]

    if v == 0:
        omegae = 2*en

    exact = (v + 1/2)*omegae
    print(f'{X.vib:2d}  {X.cm:8.2f}  {en - exact:8.2f}')

    if v in [1, 2]:
        wf = -wf

    ax.plot(R, wf*100 + X.cm)
    ax.annotate(f'  {X.vib:d}', (R[-1], X.cm), ha='left', va='center',
                fontsize='small')
    ax.plot(R, phi(X.vib, R, 10)*100 + X.cm, 'k--')

ax.plot(np.NaN, np.NaN, 'k-', label=r'PyDiatomic')
ax.plot(np.NaN, np.NaN, 'k--', label=r'analytical')
ax.plot(R, V*8065.541, ':', label='potential curve')
ax.axis(xmin=-2.5, xmax=2.5, ymin=-50, ymax=1000)
ax.legend(labelspacing=0.3, fontsize='small')
ax.set_title(r'Harmonic oscillator potential:'
             r' $\psi_v(x) = N_v H_v(x) e^{-x^2/2}$')
ax.set_ylabel(r'potential energy (cm$^{-1}$) / wavefunction $\times 100$')
ax.set_xlabel(r'$x$')

plt.savefig('figures/harmonic_oscillator.svg')
plt.show()
