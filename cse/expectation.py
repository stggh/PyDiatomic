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
    # Fix me! - needs to pythonized/vectorized
    kk = self.μ*const.e*2e-20/const.hbar**2
    e = self.cm/self._evcm

    n = self.limits[1]
    oo = self.limits[0]
    x = self.R
    dx = x[2] - x[1]

    g = np.zeros_like(x)
    x0 = np.zeros_like(x)
    x2 = np.zeros_like(x)
    v = np.zeros_like(x)
    wks = np.zeros_like(x)

    Dv = 0
    for j in np.arange(n):
        for i in np.arange(oo):
            x2[i] = 1/x[i]**2
            v[i] = self.VT[0, 0][i]*kk + self.rot*(self.rot + 1)*x2[i]
            x0[i] = self.wavefunction.T[j, 0][i]
            g[i] = x0[i]*(x2[i] - self.Bv*kk/self._evcm)

        if x[0] < 1e-10:
            v[0] = 2*v[1] - v[2]
            g[0] = 2*g[1] - g[2]

        mid = int((x[-1] - x[0])/dx/2)
        while mid > 1 and np.abs(x0[mid]) < 0.1:
            mid -= 1

        while mid > 1 and (np.abs(x0[mid-1]) < np.abs(x0[mid])):
            mid -= 1

        e *= kk 
        x1 = lideo(v, g, x0, oo, e, dx, wks, x2, mid)

        g *= x1
        Dv += simps(g, x)

    return -(Dv/kk)*self._evcm 


def lideo(v, g, x0, nip, e0, h, diag, orth, mid):
    # Fix me! - needs to pythonized/vectorized
    x1 = np.zeros_like(x0)
    mid1 = mid - 1
    nip1 = nip - 1
    h2 = h*h
    h12 = h2/12
    olap = 0

    for n in np.arange(0, nip):
        vtemp = 1/(1 - h12*(v[n] - e0))
        diag[n] = 10 - 12*vtemp
        x1[n] = h2*vtemp*g[n]
        orth[n] = x0[n]*vtemp
        olap -= h12*orth[n]*g[n]

    ort = orth[0]
    xx = x1[0]
    dd = 1/diag[0]
    for n in np.arange(1, mid):
        olap -= xx*ort*dd
        ort = orth[n] - ort*dd
        xx = x1[n] - xx*dd
        x1[n] = xx
        dd = diag[n] - dd
        diag[n] = dd
        dd = 1/dd
    orsav = ort 

    ort = orth[nip1]
    xx = x1[nip1]
    dd = 1/diag[nip1]
    k = nip1
    midp = mid + 1
    for n in np.arange(midp, nip):
        k -= 1
        olap -= xx*ort*dd
        ort = orth[k] - ort*dd
        xx = x1[k] - xx*dd
        x1[k] = xx
        dd = diag[k] - dd
        diag[k] = dd
        dd = 1/dd

    xx = (olap - ort*x1[mid1])/(orsav - diag[mid1]*ort)     
    x1[mid1] = xx
    midp = mid
    for n in np.arange(midp, nip):
        xx = (x1[n] - xx)/diag[n]
        x1[n] = xx

    k = mid1
    xx = x1[mid1]
    for n in np.arange(1, mid):
        k -= 1
        xx = (x1[k] - xx)/diag[k]
        x1[k] = xx

    for n in np.arange(nip):
        x1[n] = (x1[n] + h12*g[n])/(1 - h12*(v[n] - e0))

    if np.abs(x1[nip1]) > np.abs(x1[nip1-1]):
        k = nip1
        for n in np.arange(1, nip):
            k -= 1
            x1[k] = 0
            if np.abs(x1[k-1]) > np.abs(x1[k]):
                break

    if np.abs(x1[1]) >= np.abs(x1[0]):
        return x1

    for n in np.arange(nip1):
        x1[n] = 0
        if np.abs(x1[n+1]) > np.abs(x1[n]):
            return x1
