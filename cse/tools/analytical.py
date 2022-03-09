# -*- coding: utf-8 -*
import numpy as np

def Wei(r, re, De, Te, b, h=0):
    """ Modified Wei potential curve 
           Jai et al. J Chem Phys 137, 014101 (2012).

    Parameters
    ----------
    r : numpy 1d-array
        internuclear distance grid
    re : float
        internuclear distance at equilibrium
    De : float
        Dissociation energy
    Te : float
        equilibrium energy (potential minimum)
    b : float
        anharmonicity parameter
    h : float
        |h| < 1, h=0 gives a Morse potential curve

    Returns
    -------
    potential_curve : numpy 1d-array
        potential energy curve

    """

    if abs(h) > 1:
        raise SyetemExit(f'Wei(r, re, De, Te, b, h): error h={h}, but |h| < 1')
        
    ebre = np.exp(b*re)
    ebr = np.exp(b*r)

    return De*(1 - ebre*(1 - h)/(ebr - h*ebre))**2 + Te


def Morse(r, re, De, Te, beta):
    """Morse potential energy curve.

    Parameters
    ----------
    r : numpy 1d-array
        internuclear distance grid
    re : float
        internuclear distance at equilibrium
    De : float
        Dissociation energy
    Te : float
        equilibrium energy (potential minimum)
    beta : float
        anharmonicity parameter

    Returns
    -------
    potential_curve : numpy 1d-array
        potential energy curve

    """

    return Wei(r, re, De, Te, beta, h=0)


def Julienne(r, mx, rx, vx, voo):
    """Julienne (and Krauss) dissociative potential energy curve.

    math:
          v(r) = v_x \exp[-(m_x/v_x)(r - r_e)] + v_\infy

    Eq. (45) J. Mol. Spect. 56, 270-308 (1975)

    Parameters
    ----------
    r : numpy 1d-array
        internuclear distance grid.
    rx : float
        crossing point with a bound curve.
    mx : float
        slope at crossing point (rx).
    vx : float
        energy at crossing point.
    voo : float
        disociation limit energy.

    Returns
    -------
    potential_curve: numpy 1d-array
        potenial energy curve
    """

    return vx*np.exp(-(mx/vx)*(r-rx)) + voo
