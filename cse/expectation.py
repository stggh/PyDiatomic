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
π = np.pi
CONST = 2*(π*const.e*a0)**2*1.0e4/const.epsilon_0/3
FCONST = 4*π*const.m_e*const.c*100*a0*a0/const.hbar/3


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

    ReX = np.zeros((oo, nopen, 1))
    ImX = np.zeros((oo, nopen, 1))
    for i in range(oo):
        ReX[i] = (Xs.dipolemoment[i] @ Re[i]).T @ wfi[i]
        ImX[i] = (Xs.dipolemoment[i] @ Im[i]).T @ wfi[i]

    xsp = np.zeros(n)  # n > nopen = max size of xs array
    for j in range(nopen):  # nopen >= 1
        Rx = simps(ReX[:, j, 0], Xs.us.R)
        Ix = simps(ImX[:, j, 0], Xs.us.R)
        xsp[j] = Rx**2 + Ix**2

    if np.any(Xs.us.openchann):
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
    Xs.us.solve(en)
    if Xs.us.openchann.size == 1 and Xs.us.openchann == 0:
        # bound upperstates => reset transition energy
        wavenumber = Xs.us.cm - Xs.gs.cm
    xsp = cross_section(wavenumber, Xs)
    return xsp, wavenumber 


def xs_vs_wav(Xs):
    """ multiprocessor pool function.

    """
    pool = multiprocessing.Pool()
    func = partial(xs, Xs)

    xsp = pool.map(func, Xs.wavenumber)
    pool.close()
    pool.join()

    return xsp


def Bv(Cse):
    """ Bv rotational constant from expectation <v|1/R^2|v>.

    """
    R = Cse.R
    wavefunction = Cse.wavefunction[:, 0, 0]
    μ = Cse.μ

    ex = simps((wavefunction/R)**2, R)
    return ex*const.hbar*1.0e18/(4*π*const.c*μ)


def Dv(self):
    """ Dv rotational constant using Hudson algorithm:
        Dv = <v| Bv - H'| v(1)>.
        Solve the linear inhomogeneous differential equation
        d^2chi/dR^2 = V(R)chi(R) + g(R).
        JM Hutson J. Phys. B14 851-857 (1981) doi:10.1088/0022-3700/14/5/018

    """
    kk = self.μ*const.e*2e-20/const.hbar**2
    e = self.cm*kk/self._evcm

    n = self.limits[1]
    oo = self.limits[0]
    dR = self.R[2] - self.R[1]
    R2 = 1/self.R**2

    wks = np.zeros_like(R2)

    Dv = 0
    for j in np.arange(n):
        v = self.VT[j, j]*kk + self.rot*(self.rot + 1)*R2
        R0 = self.wavefunction.T[j, 0]
        g = R0*(R2 - self.Bv*kk/self._evcm)

        mid = int((self.R[-1] - self.R[0])/dR/2)
        while mid > 1 and np.abs(R0[mid]) < 0.1:
            mid -= 1

        while mid > 1 and (np.abs(R0[mid-1]) < np.abs(R0[mid])):
            mid -= 1

        R1 = lideo(v, g, R0, oo, e, dR, wks, R2, mid)

        g *= R1
        Dv += simps(g, self.R)

    return -(Dv/kk)*self._evcm 


def lideo(v, g, R0, oo, e0, dR, diag, orth, mid):
    mid1 = mid - 1
    dR2 = dR**2
    dR12 = dR2/12

    vtemp = 1/(1 - dR12*(v - e0))
    diag = 10 - 12*vtemp
    R1 = dR2*vtemp*g
    orth = R0*vtemp
    olap = -(dR12*orth*g).sum()

    ort = orth[0]
    xx = R1[0]
    dd = 1/diag[0]
    for n in np.arange(1, mid):
        olap -= xx*ort*dd
        ort = orth[n] - ort*dd
        xx = R1[n] - xx*dd
        R1[n] = xx
        dd = diag[n] - dd
        diag[n] = dd
        dd = 1/dd
    orsav = ort 

    ort = orth[-2]
    xx = R1[-2]
    dd = 1/diag[-2]
    for n in np.arange(oo-2, mid-1, -1):
        olap -= xx*ort*dd
        ort = orth[n] - ort*dd
        xx = R1[n] - xx*dd
        R1[n] = xx
        dd = diag[n] - dd
        diag[n] = dd
        dd = 1/dd

    xx = (olap - ort*R1[mid1])/(orsav - diag[mid1]*ort)     
    R1[mid1] = xx
    for n in np.arange(mid, oo):
        xx = (R1[n] - xx)/diag[n]
        R1[n] = xx

    xx = R1[mid1]
    for n in np.arange(mid1-1, 0, -1):
        xx = (R1[n] - xx)/diag[n]
        R1[n] = xx

    R1 = (R1 + dR12*g)/(1 - dR12*(v - e0))

    if np.abs(R1[-1]) > np.abs(R1[-2]):
        for n in np.range(oo-2, 1, -1):
            R1[n] = 0
            if np.abs(R1[n-1]) > np.abs(R1[n]):
                break

    if np.abs(R1[1]) >= np.abs(R1[0]):
        return R1

    for n in np.arange(oo-1):
        R1[n] = 0
        if np.abs(R1[n+1]) > np.abs(R1[n]):
            return R1
