# -*- coding: utf-8 -*
###############################################################################
# rkr.py
# Rydberg-Klein-Rees potential energy curve from spectroscopic constants
# Gv Bv
# Stephen.Gibson@anu.edu.au - May 2016
###############################################################################

import numpy as np
import scipy.constants as const
from scipy.interpolate import splrep, splev

def rkr(mu, vv, Gv, Bv, De, limb='L', dv=0.1,
        Rgrid=np.arange(0.005, 10.004, 0.005), verbose=True):
    """ Rydberg-Klien-Rees potential energy curve.

    Parameters
    ---------
    mu : float
        readuced mass in amu
    vv : numpy 1d array of floats
        vibrational quantum number for Gv, Bv
    Gv : numpy 1d array of floats
        vibrational constants at vv in cm-1
    Bv : numpy 1d array of floats
        rotational constants at vv in cm-1
    De : float
        dissociation energy in cm-1
    limb : str
        analytical function to extrapolate outer limb to the dissociation limit
        'L' = LeRoy otherwise Morse
        note, inner limb is Morse
    dv : float
        evaluate the turning points at dv increments
    Rgrid : numpy 1d array of floats
        radial grid on which to interpolate and extrapolate the potential
        energy curve
    verbose: boolean
        print results of calculated turning points and potential curve
        extension parameters
        
    Returns
    -------
    Rgrid : numpy 1d array of float
        grid for internuclear distance
    PEC : numpy 1d array of float
        potential energy curve
    RTP : numpy 1d array of float
        internuclear distance at turning point
    PTP : numpy 1d array of float
        potential energy at turning point

    """

    Rmin, Rmax, E = turning_points(mu, vv, Gv, Bv, dv, verbose=verbose)

    PEC, RTP, PTP = formPEC(Rgrid, Rmin, Rmax, E, De, limb, verbose=verbose)

    return Rgrid, PEC, RTP, PTP


# interpolate Gv, Bv
def G(v, gsp):
    return splev(v, gsp, der=0)

def B(v, bsp):
    return splev(v, bsp, der=0)


# RKR integrals
def fg_integral(v, gsp, bsp, func):
    # Gaussian integration with weight function (see STG thesis, p58)
    # or Wikipedia https://en.wikipedia.org/wiki/Gaussian_quadrature
    xi = np.array([-0.9578284203, -0.7844439484, -0.4986347571, -0.1412716403,
                    0.2364578932, 0.5804412628, 0.8413988804, 0.9819452459])
    hi = np.array([0.0767987527, 0.1760795557, 0.2691489156, 0.3525039628,
                   0.4231213525, 0.4784468878, 0.5164804522, 0.5358472454])
    dv = v + 1/2
    e = G(v, gsp)
    sumi = 0
    for l, x in enumerate(xi):
        vd = (v - 1/2 + x*dv)/2
        sumi += hi[l]*np.sqrt(1 - x)*func(vd, bsp)/np.sqrt(e - G(vd, gsp))
    return sumi*dv/2


def turning_points(mu, vv, Gv, Bv, dv=0.1, verbose=True):
    DD = np.sqrt(const.h/(8*mu*const.m_u*const.c*100))*1.0e10/np.pi
    # Gv spline
    gsp = splrep(vv, Gv, s=0)
    # Bv spline
    bsp = splrep(vv, Bv, s=0)

    # vibrational QN at which to evaluate turning points
    V = np.arange(-1/2, vv[-1]+dv/2, dv)
    # offset the point at v=-0.5, the bottom of the well
    V[0] = -1/2 + 0.0001
    Rmin = []
    Rmax = []
    E = []
    # compute turning points using RKR method
    if verbose:
        print(u"RKR:  v    Rmin(A)   Rmax(A)     E(cm-1)")
        vi = 0

    for vib in V:
        E.append(G(vib, gsp))    # energy of vibrational level
        ff = fg_integral(vib, gsp, bsp, lambda x, y: 1)
        gg = fg_integral(vib, gsp, bsp, B)
        fg = np.sqrt(ff/gg + ff**2)
        Rmin.append((fg - ff)*DD)  # turning points
        Rmax.append((fg + ff)*DD)
        if verbose:
            if np.isclose(vib, vi):
                print(f'     {vi:2d}   {Rmin[-1]:7.4f}   {Rmax[-1]:7.4f}'
                      f'    {np.float(E[-1]):9.2f}')
                vi += 1

    return Rmin, Rmax, E


def formPEC(R, Rmin, Rmax, E, De, limb, verbose=True):
    evcm = const.e/(const.c*const.h*100)  # converts cm-1 to eV

    # combine Rmin with Rmax to form PEC
    Re = (Rmin[0] + Rmax[0])/2
    if verbose:
        print(u"RKR: Re = {:g}".format(Re))

    RTP = np.append(Rmin[::-1], Rmax, 0)  # radial positions of turning-points
    PTP = np.append(E[::-1], E, 0)  # potential energy at turning point

    # interpolate
    psp = splrep(RTP, PTP, s=0)
    # Interpolate RKR curve to this grid
    PEC = splev(R, psp, der=0)

    # extrapolate using analytical function
    inner_limb_Morse(R, PEC, RTP, PTP, Re, De, verbose=verbose)
    if limb == 'L':
        outer_limb_LeRoy(R, PEC, RTP, PTP, De, verbose=verbose)
    else:
        outer_limb_Morse(R, PEC, RTP, PTP, De, Re, verbose=verbose)

    PTP /= evcm
    PEC /= evcm  # convert to eV

    return PEC, RTP, PTP


# analytical functions
def inner_limb_Morse(R, P, RTP, PTP, Re, De, verbose=True):
    ln0 = np.log(1 + np.sqrt(PTP[0]/De))
    beta = ln0/(Re - RTP[0])

    subR = R < RTP[0]
    P[subR] = De*( 1 - np.exp(-beta*(R[subR]-Re)) )**2

    if verbose:
        print('RKR: Inner limb  De[1-exp(beta*(Re-R))]^2')
        print(f'RKR:  {R[0]:g}-{RTP[0]:g}A   {De:g}(De)  {Re:g}(Re)  '
              f'{beta:g}(beta)')


def outer_limb_Morse(R, P, RTP, PTP, De, Re, verbose=True):
    l1 = np.log(1 - np.sqrt(PTP[-1]/De))
    l2 = np.log(1 - np.sqrt(PTP[-2]/De))
    Re = (l2*RTP[-1]-l1*RTP[-2])/(l2-l1)
    beta = l1/(Re - RTP[-1])

    subR = R > RTP[-1]
    P[subR] = De*( 1 - np.exp(-beta*(R[subR]-Re)) )**2

    if verbose:
        print('RKR: Outer limb  De[1-exp(beta*(Re-R))]^2')
        print(f'RKR:   {RTP[-1]:g}-{R[-1]:g}A   {De:g}(De)  {Re:g}(Re)  '
              f'{beta:g}(beta)')


def outer_limb_LeRoy(R, P, RTP, PTP, De, verbose=True):
    n = np.log((De - PTP[-1])/(De - PTP[-2]))/np.log(RTP[-2]/RTP[-1])
    Cn = (De - PTP[-1])*RTP[-1]**n

    subR = R > RTP[-1]
    P[subR] = De - Cn/R[subR]**n

    if verbose:
        print('RKR: Outer limb  De - Cn/R^n')
        print(f'RKR:  {RTP[-1]:g}-{R[-1]:g}A   {De:g}(De)  {n:g}(n)  '
              f'{Cn:g}(Cn)')
