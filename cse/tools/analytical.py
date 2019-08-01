# -*- coding: utf-8 -*
import numpy as np


def Morse(r, re, De, Te, beta):
    """Morse potential energy curve.

    Parameters
    ----------
    r : numpy 1d array
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

    return De*(1-np.exp(-beta*(r-re)))**2 + Te
