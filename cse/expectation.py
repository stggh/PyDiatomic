# -*- coding: utf-8 -*-
import numpy as np
import scipy.constants as const

from scipy import linalg
from scipy.optimize import leastsq
from scipy.integrate.quadrature import simps
import itertools

import multiprocessing
from functools import partial

from . import johnson

"""
  expectation values - cross section, oscillator strengths,
                       spectroscopic constanst ...

  Stephen.Gibson@anu.edu.au
  February 2016
"""

a0 = const.physical_constants["Bohr radius"][0]
CONST = 2*(np.pi*const.e*a0)**2*1.0e4/const.epsilon_0/3
FCONST = 4*np.pi*const.m_e*const.c*100*a0*a0/const.hbar/3

def cross_section(wavenumber, Xs):
    """ photodissociation cross section |<f|M|i>|^2.

    Parameters
    ----------
    wavenumber : float
        wavenumber at which to evaluate the cross section
    wfu : numpy 3d array oo, n, nopen
        wavefunction array for the upper coupled states
    wfi : numpy 3d array oo, ni, 1
        wavefunction array for the initial coupled states
    dipolemoment : numpy array
        transition moments between initial and final PECs

    Returns
    -------
    cross section : numpy array
        photodissociation cross section for each open channel

    """

    wfi = Xs.gs.wavefunction
    wfu = Xs.us.wavefunction
    oo, n, nopen = wfu.shape

    Re = wfu.real
    Im = wfu.imag

    ReX = np.zeros((oo, nopen))
    ImX = np.zeros((oo, nopen))
    for i in range(oo):
        ReX[i] = (Xs.dipolemoment[i, 0] @ Re[i])*wfi[i]
        ImX[i] = (Xs.dipolemoment[i, 0] @ Im[i])*wfi[i]

    xsp = np.zeros(n)  # n > nopen = max size of xs array 
    for j in range(nopen):  # nopen >= 1
        Rx = simps(ReX[:, j], Xs.us.R)
        Ix = simps(ImX[:, j], Xs.us.R)
        xsp[j] = Rx**2 + Ix**2

    if np.any(Xs.openchann):
        # cross section
        Xs.xs = np.array(xsp)*wavenumber*CONST*1.0e-8
    else:
        # oscillator strength
        Xs.xs = np.array(xsp)*wavenumber*FCONST
    return Xs.xs


def xs(Xs, wavenumber):
    """ solve CSE of upper coupled states for the transition energy.

    """
    dE = wavenumber/8065.541  # convert to eV
    en = Xs.gs.energy + dE
    Xs.us.wavefunction, Xs.us.energy, Xs.openchann = johnson.solveCSE(en, 
                                      Xs.us.rot, Xs.us.mu, Xs.us.R, Xs.us.VT)
    xsp = cross_section(wavenumber, Xs)
    hlf = honl(Xs)
    return xsp*hlf  #  (wavenumber.shape, n)


def xs_vs_wav(Xs):
    """ multiprocessor pool function.

    """
    pool = multiprocessing.Pool()
    func = partial(xs, Xs)

    xsp = pool.map(func, Xs.wavenumber)
    pool.close()
    pool.join()

    return np.array(xsp)


def honl(Xs):
    # Honl-London factor
    hfl = 1.0
    if Xs.honl:
        Jd = Xs.us.rot
        Jdd = Xs.gs.rot
        Od = Xs.us.Omega
        Odd = Xs.gs.Omega
        hfl = (2*Jd + 1) * N(wigner_3j(Jd, 1, Jdd, -Od, Od-Odd, Odd))**2
    return hfl 


def Bv(R, wavefunction, mu):
    """ Bv rotational constant from expectation <v|1/R^2|v>.

    """
    ex = simps((wavefunction/R)**2, R)
    return ex*const.hbar*1.0e18/(4*np.pi*const.c*mu)
