# -*- coding: utf-8 -*-
import numpy as np
import scipy.constants as const

from scipy import linalg
from scipy.optimize import leastsq
from scipy.integrate.quadrature import simps

import multiprocessing
from functools import partial

from . import johnson

u"""
  expectation values - cross section, oscillator strengths, 
                       spectroscopic constanst ...

  Stephen.Gibson@anu.edu.au
  February 2016
"""


a0 = const.physical_constants["Bohr radius"][0]
CONST = 2*(np.pi*const.e*a0)**2*1.0e4/3/const.epsilon_0

def cross_section(wavenumber, wfu, wfi, R, dipolemoment):
    u""" photodissociation cross section |<f|M|i>|^2.

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

    oo, n, nopen = wfu.shape

    Re = wfu.real
    Im = wfu.imag

    ReX = np.zeros((oo, nopen))
    ImX = np.zeros((oo, nopen))
    for i in range(oo):
        ReX[i] = (dipolemoment[i] @ Re[i])*wfi[i]
        ImX[i] = (dipolemoment[i] @ Im[i])*wfi[i]

    xsp = []
    for j in range(nopen):
        Rx = simps(ReX[:,j], R)
        Ix = simps(ImX[:,j], R)
        xx = Rx**2 + Ix**2
        xs = xx*CONST*wavenumber*1.0e-8
        xsp.append(xs)
    
    return xsp


def xs(dipolemoment, ei, rot, mu, R, VT, wfi, wavenumber):
    u""" solve CSE of upper coupled states for the transition energy.

    """
    dE = wavenumber/8065.541
    en = ei + dE
    wfu, eu = johnson.solveCSE(en, rot, mu, R, VT)
    xsp = cross_section(wavenumber, wfu, wfi, R, dipolemoment)
    return xsp


def xs_vs_wav(wavenumber, dipolemoment, ei, rot, mu, R, VT, wfi):
    u""" mulitprocessor pool function.

    """
    pool = multiprocessing.Pool()
    func = partial(xs, dipolemoment, ei, rot, mu, R, VT, wfi)

    xsp = pool.map(func, wavenumber)
    xsp = np.array(xsp)
    pool.close()
    pool.join()
   
    return np.array(xsp)


def Bv(R, wavefunction, mu):
    u""" Bv rotational constant from expectation <v|1/R^2|v>.
 
    """
    ex = simps((wavefunction/R)**2, R)
    return ex*const.hbar*1.0e18/(4*np.pi*const.c*mu)
