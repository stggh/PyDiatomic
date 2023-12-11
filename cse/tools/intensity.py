import numpy as np
import scipy.constants as const
from scipy.special import factorial
from scipy.interpolate import InterpolatedUnivariateSpline, splrep, splev

def bandosc(transition_instance, vmax=None, verbose=True):
    """ Band oscillator strength and density of states

    Parameters
    ----------
    transition_instance : Cse Transition class 
    vmax : int
        highest vibrational level for which to evaluate the oscillator strength
    verbose : bool
        print the band oscillator strengths

    Returns
    -------
    vib : numpy array of ints
        vibrational quantum numbers
    bands : numpy array of floats
        transition energies in cm⁻¹)
    fosc : numpy array of floats
        band oscillator strengths
    dvdE : numpy array of floats
        density of states dv/dE

    """

    SX = transition_instance
    S = SX.us  # final state
    X = SX.gs  # initial state

    if verbose:
        print('\nBand oscillator stengths')
        print(f' {S.statelabel[0]} <- {X.statelabel[0]}:')

    try:
        Xzpe = X.cm
    except:
        X.solve(X.VT[0, 0].min()*X._evcm+500)
        Xzpe = X.results[0][0]  # Tv=0

    S.levels(vmax)

    vib = []
    bands = []
    for v, (Tv, *_) in S.results.items():
        vib.append(v)
        bands.append(Tv - Xzpe)

    spl = InterpolatedUnivariateSpline(bands, vib, k=1)
    dvdE = spl.derivative()(bands)

    SX.calculate_xs(transition_energy=bands)
    fosc = SX.xs[:, 0]

    if verbose:
        print(' v      fosc          ratio')
        for v, f in zip(vib, fosc):
            print(f'{v:2d}   {f:10.3e}    ', end='')
            if (ratio := f/fosc[0]) > 1e-3:
                print(f'   {ratio:5.3f}')
            else:
                print(f'   {ratio:5.1e}')
        print()

    return vib, bands, fosc, dvdE


def Boltzmann(en, J, T):
    """ Boltzmann factor  (2J+1) exp(-E/kT).

    Parameters
    ----------
    en : float
        energy in cm-1 of level
    J : int
        rotational quantum number
    T : float
        temperature
    
    Returns
    -------
    Boltzmann factor : float
    """

    return (2*J + 1)*np.exp(-en*const.h*const.c*100/const.k/T)


def gfactor(Λd, Λdd):
    """ (2 - δ₀Λ'δ₀Λ")/(2 - δ₀Λ").

    """
    return (2 - np.kron(0, Λd)*np.kron(0, Λdd))/(2 - np.kron(0, Λdd))


def honl(Jd, Jdd, Ωd, Ωdd, q=None):
    """ Honl-London factor.

    Parameters
    ----------
    Jd : int
        upperstate total angular momentum quantum number
    Jdd : int
        lowerstate total angular momentum quantum number
    Ωd : int
        upperstate Omega, projection of Jd
    Ωdd : int
        lowerstate Omega, projection of Jdd
    q : int
        0 - parallel or ±1 perpendicular transitions

    Returns
    -------
    Honl-London factor : float
    """
     
    if q is None:
        q = Ωd - Ωdd
    # (2J'+1)(2J"+1)*Wig**2/(2J"+1)
    return (2*Jd+1)*Wigner3j(Jd, 1, Jdd, -Ωd, q, Ωdd)**2


def Wigner3j(j1, j2, j3, m1, m2, m3):
#======================================================================
# Wigner3j.m by David Terr, Raytheon, 6-17-04
#
# Compute the Wigner 3j symbol using the Racah formula [1]. 
#
# Usage: 
# from wigner import Wigner3j
# wigner = Wigner3j(j1,j2,j3,m1,m2,m3)
#
#  / j1 j2 j3 \
#  |          |  
#  \ m1 m2 m3 /
#
# Reference: Wigner 3j-Symbol entry of Eric Weinstein's Mathworld: 
# http://mathworld.wolfram.com/Wigner3j-Symbol.html
#======================================================================

    # Error checking
    if  2*j1 != np.floor(2*j1) or 2*j2 != np.floor(2*j2) or \
        2*j3 != np.floor(2*j3) or 2*m1 != np.floor(2*m1) or \
        2*m2 != np.floor(2*m2) or 2*m3 != np.floor(2*m3):
        #print('All arguments must be integers or half-integers.')
        return 0

    # Additional check if the sum of the second row equals zero
    if m1 + m2 + m3 != 0:
        # print('3j-Symbol unphysical')
        return 0

    if j1 - m1 != np.floor(j1 - m1):
        # print('2*j1 and 2*m1 must have the same parity')
        return 0
    
    if j2 - m2 != np.floor(j2 - m2):
        # print('2*j2 and 2*m2 must have the same parity')
        return; 0

    if j3 - m3 != np.floor(j3 - m3):
        # print('2*j3 and 2*m3 must have the same parity')
        return 0
    
    if j3 > j1 + j2  or j3 < np.abs(j1 - j2):
        # print('j3 is out of bounds')
        return 0

    if np.abs(m1) > j1:
        # print('m1 is out of bounds')
        return 0

    if np.abs(m2) > j2:
        # print('m2 is out of bounds')
        return 0 

    if np.abs(m3) > j3:
        # print('m3 is out of bounds')
        return 0

    t1 = j2 - m1 - j3
    t2 = j1 + m2 - j3
    t3 = j1 + j2 - j3
    t4 = j1 - m1
    t5 = j2 + m2

    tmin = max(0, max(t1, t2))
    tmax = min(t3, min(t4, t5) )
    tvec = np.arange(tmin, tmax+1, 1)

    wigner = 0

    for t in tvec:
        wigner += (-1)**t / (factorial(t) * factorial(t-t1) * 
                             factorial(t-t2) * factorial(t3-t) * 
                             factorial(t4-t) * factorial(t5-t))

    return wigner * (-1.)**(j1-j2-m3) * np.sqrt(factorial(j1+j2-j3) * 
           factorial(j1-j2+j3) * factorial(-j1+j2+j3) / 
           factorial(j1+j2+j3+1) * factorial(j1+m1) * 
           factorial(j1-m1) * factorial(j2+m2) * factorial(j2-m2) *
           factorial(j3+m3) * factorial(j3-m3) )
