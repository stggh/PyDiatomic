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
from scipy.optimize import curve_fit

def rkr(mu, vv, Gv, Bv, De, limb='L', dv=0.1,
        Rgrid=np.arange(0.005, 10.004, 0.005), verbose=True):
    """ Rydberg-Klien-Rees potential energy curve.

    Parameters
    ---------
    mu : float
        readuced mass in amu
    vv : numpy 1d array of floats
        vibrational quantum number for Gv, Bv
    Gv : (= Te+Gv) numpy 1d array of floats
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
    # offset the point at v=-1/2, the bottom of the well
    V[0] = -1/2 + 0.0001

    # compute turning points using RKR method
    E = G(V, gsp)
    ff = fg_integral(V, gsp, bsp, lambda x, y: 1)
    gg = fg_integral(V, gsp, bsp, B)
    fg = np.sqrt(ff/gg + ff**2)

    Rmin = (fg - ff)*DD  # turning points
    Rmax = (fg + ff)*DD

    if verbose:
        vint = np.mod(V, 1) > 0.99
        
        print(u"RKR:  v    Rmin(A)   Rmax(A)     E(cm-1)")
        for vib, rm, rx, e in zip(V[vint], Rmin[vint], Rmax[vint], E[vint]):
            if vib >= 0: 
                print(f'    {int(vib):2d}   {rm:7.4f}   {rx:7.4f}'
                      f'{np.float(e):9.2f}')
            else:
                print(f'   {"-1/2":s}  {rm:7.4f}   {rx:7.4f}'
                      f'{np.float(e):9.2f}')

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

    bound = np.logical_and(R > RTP[0], R < RTP[-1])
    Te = PEC[bound].min()

    # extrapolate using analytical function
    inner_limb_Morse(R, PEC, RTP, PTP, Re, De, Te, verbose=verbose)
    if limb == 'L':
        outer_limb_LeRoy(R, PEC, RTP, PTP, De, Te, verbose=verbose)
    else:
        outer_limb_Morse(R, PEC, RTP, PTP, De, Re, Te, verbose=verbose)

    PTP /= evcm
    PEC /= evcm  # convert to eV

    return PEC, RTP, PTP

def Morse(x, beta, De, Re, Te):
    return De*(1 - np.exp(-beta*(x-Re)))**2 + Te

# analytical functions
def inner_limb_Morse(R, P, RTP, PTP, Re, De, Te, verbose=True):
    # V(r) = De[1 - exp(-β(R-Re))]² + Te

    # evaluate β based on inner-most turning point
    ln0 = np.log(1 + np.sqrt((PTP[0] - Te)/De))
    beta = ln0/(Re - RTP[0])

    # fit Morse to inner turning points
    inner = len(PTP) // 2
    popt, pcov = curve_fit(Morse, RTP[:inner], PTP[:inner], 
                           p0=[beta, De, Re, Te])
    err = np.sqrt(np.diag(pcov))

    subR = R < RTP[0]
    P[subR] = Morse(R[subR], *popt)

    if verbose:
        print('\nRKR: Inner limb  De[1-exp(β(Re-R))]² + Te')
        for par, val, er in zip(['β', 'De', 'Re', 'Te'], popt, err):
            print(f'RKR:  {par:3s}  {val:12.2f}±{er:.2f}')


def outer_limb_Morse(R, P, RTP, PTP, De, Re, Te, verbose=True):
    # V(r) = De[1 - exp(-β(R-Re))]² + Te
    l1 = np.log(1 - np.sqrt((PTP[-1] - Te)/De))
    l2 = np.log(1 - np.sqrt((PTP[-2] - Te)/De))
    Re = (l2*RTP[-1] - l1*RTP[-2])/(l2 - l1)
    beta = l1/(Re - RTP[-1])

    # fit Morse to outer turning points
    outer = len(PTP) // 2
    popt, pcov = curve_fit(Morse, RTP[outer:], PTP[outer:], 
                           p0=[beta, De, Re, Te])
    err = np.sqrt(np.diag(pcov))

    subR = R > RTP[-1]
    P[subR] = Morse(R[subR], *popt) 

    if verbose:
        print('\nRKR: Outer limb  De[1-exp(β(Re-R))]² + Te')
        for par, val, er in zip(['β', 'De', 'Re', 'Te'], popt, err):
            print(f'RKR:  {par:3s}  {val:12.2f}±{er:.2f}')


def outer_limb_LeRoy(R, P, RTP, PTP, De, Te, verbose=True):
    def LeRoy(x, De, Cn, n, Te):
        return De - Cn/x**n + Te

    n = int(np.log((Te + De - PTP[-1])/(Te + De - PTP[-2]))\
        /np.log(RTP[-2]/RTP[-1]))
    Cn = (Te + De - PTP[-1])*RTP[-1]**n

    # fit long-range potential curve to outer turning points
    outer = len(PTP)*2 // 3   # choose last 1/3 of points
    popt, pcov = curve_fit(lambda x, De, Cn: LeRoy(x, De, Cn, n, Te), 
                           RTP[outer:], PTP[outer:], p0=[De, Cn])
    err = np.sqrt(np.diag(pcov))

    subR = R > RTP[-1]
    P[subR] = LeRoy(R[subR], De, Cn, n, Te)

    if verbose:
        print('\nRKR: Outer limb  De - Cn/R^n')
        for par, val, er in zip(['De', 'Cn'], popt, err):
            print(f'RKR:  {par:3s}  {val:12.2f}±{er:.2f}')
        print(f'RKR: n = {n}, Te = {Te:8.2f}')
