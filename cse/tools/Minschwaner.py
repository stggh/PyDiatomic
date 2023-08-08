import numpy as np
import cse
import os

def xs(a, b, c, T):
    delta = ((T-100)/10)**2
    return (a*delta**2 + b*delta + c)*1e-20

def O2(T=300):
    """ K. Minschwaner, G. P. Anderson, L. A. Hall, and K. Yoshino
        'Polynomial coefficients for calculating O2 Schumann-Runge
        cross sections at 0.5 cm-1 resolution' JGR 97, D9, 10103-10108 (1992).
    """

    if T <= 190:
        fn = 'fitcoef_cold.txt'
    elif T <= 280:
        fn = 'fitcoef_mid.txt'
    else:
        fn = 'fitcoef_hot.txt'
    coeff = os.path.join('tools', fn)
    coeff_file = os.path.join(cse.__path__[0], coeff)

    wn, a, b, c = np.loadtxt(coeff_file, usecols=(0, 1, 2, 3), unpack=True)

    xsT = xs(a, b, c, T=T)

    return wn, xsT


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    T = 300
    wn, xsT = O2(T)
    Yoshino = np.loadtxt('../../examples/O2/data/Harvard/o2wb12x0.xsc',
                         unpack=True)

    plt.semilogy(wn, xsT)
    plt.title(f'{T:d} K')
    plt.plot(*Yoshino)
    plt.xlabel(r'wavenumber (cm$^{-1}$)')
    plt.ylabel(r'cross section (cm$^2$)')
    plt.show()
