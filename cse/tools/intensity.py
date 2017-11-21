import numpy as np
import scipy.constants as const
from sympy.physics.wigner import wigner_3j

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

def honl(Jd, Jdd, Od, Odd):
    """ Honl-London factor.

    Parameters
    ----------
    Jd : int
        upperstate total angular momentum quantum number
    Jdd : int
        lowerstate total angular momentum quantum number
    Od : int
        upperstate Omega, projection of Jd
    Odd : int
        lowerstate Omega, projection of Jdd

    Returns
    -------
    Honl-London factor : float
    """
     
    return (2*Jdd + 1)*np.float(wigner_3j(Jd, 1, Jdd, -Od, 0, Odd))**2
