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

def rkr(μ, vv, Gv, Bv, De, Voo=0, limb='L', dv=0.1,
        Rgrid=np.arange(0.005, 10.004, 0.005), ineV=False, verbose=True):
    """ Rydberg-Klien-Rees potential energy curve.

    Parameters
    ---------
    μ : float
        reduced mass in amu
    vv : numpy 1d array of floats
        vibrational quantum numbers for Gv, Bv
    Gv : (= Te+Gv) numpy 1d array of floats
        vibrational constants at vv in cm-1
    Bv : numpy 1d array of floats
        rotational constants at vv in cm-1
    De : float
        depth of potential energy curve well in cm-1
    Voo: float
        dissociation energy in cm-1
    limb : str
        analytical function to extrapolate outer limb to the dissociation limit
        'L' = LeRoy otherwise Morse. Note, inner limb is Morse
    dv : float
        evaluate the turning points at dv increments
    Rgrid : numpy 1d array of floats
        radial grid on which to interpolate and extrapolate the potential
        energy curve
    ineV : boolean
        return potential curve energy in eV, rather than cm-1
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
    if Voo < De:
        Voo = De + Gv[0]

    Rmin, Rmax, E = turning_points(μ, vv, Gv, Bv, dv, verbose=verbose)

    PEC, RTP, PTP = formPEC(Rgrid, Rmin, Rmax, E, De, Voo, limb, ineV,
                            verbose=verbose)

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


def turning_points(μ, vv, Gv, Bv, dv=0.1, verbose=True):
    DD = np.sqrt(const.h/(8*μ*const.m_u*const.c*100))*1.0e10/np.pi
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
                print(f'     {int(vib):2d}   {rm:7.4f}   {rx:7.4f}'
                      f'    {np.float(e):9.2f}')
            else:
                print(f'    {"-1/2":s}  {rm:7.4f}   {rx:7.4f}'
                      f'    {np.float(e):9.2f}')

    return Rmin, Rmax, E


def formPEC(R, Rmin, Rmax, E, De, Voo, limb, ineV=False, verbose=True):
    evcm = const.e/(const.c*const.h*100)  # converts cm-1 to eV

    # combine Rmin with Rmax to form PEC
    Re = (Rmin[0] + Rmax[0])/2
    if verbose:
        print(f'RKR: Re = {Re:g}')

    RTP = np.append(Rmin[::-1], Rmax, 0)  # radial positions of turning-points
    PTP = np.append(E[::-1], E, 0)  # potential energy at turning point

    # interpolate
    psp = splrep(RTP, PTP, s=0)
    # Interpolate RKR curve to this grid
    PEC = splev(R, psp, der=0)

    bound = np.logical_and(R > RTP[0], R < RTP[-1])
    Te = PEC[bound].min()

    # extrapolate using analytical function
    inner_limb_Morse(R, PEC, RTP, PTP, Re, De, Voo, verbose=verbose)
    if limb == 'L':
        outer_limb_LeRoy(R, PEC, RTP, PTP, De, Voo, verbose=verbose)
    else:
        outer_limb_Morse(R, PEC, RTP, PTP, Re, De, Voo, verbose=verbose)

    if ineV:  # convert to eV
        PTP /= evcm
        PEC /= evcm

    return PEC, RTP, PTP

def Morse(x, β, Re, De, Voo):
    # V(r) = De[1 - exp(-β(R-Re))]² + Te
    Te = Voo - De
    return De*(1 - np.exp(-β*(x-Re)))**2 + Te

# analytical functions
def inner_limb_Morse(R, P, RTP, PTP, Re, De, Voo, inner=None, verbose=True):
    # V(r) = De[1 - exp(-β(R-Re))]² + Te
    Te = Voo - De

    # evaluate β based on inner-most turning point
    ln0 = np.log(1 + np.sqrt((PTP[0] - Te)/De))
    β = ln0/(Re - RTP[0])

    # fit Morse to inner turning points - adjusting β, Re, De; fixed Voo
    if not inner:
        inner = len(PTP) // 2
    popt, pcov = curve_fit(lambda x, β, Re, De: Morse(x, β, Re, De, Voo),
                           RTP[:inner], PTP[:inner], p0=[β, Re, De])
    err = np.sqrt(np.diag(pcov))

    subR = R < RTP[0]
    P[subR] = Morse(R[subR], *popt, Voo)

    if verbose:
        print('\nRKR: Inner limb  Morse: De[1-exp(β(Re-R))]² + Te')
        for par, val, er in zip(['β', 'Re', 'De', 'Voo'], popt, err):
            print(f'RKR:  {par:3s}  {val:12.2f}±{er:.2f}')
        print(f'RKR:  Te   {Voo - popt[-1]:12.2f} cm-1')


def outer_limb_Morse(R, P, RTP, PTP, Re, De, Voo, verbose=True):
    # V(r) = De[1 - exp(-β(R-Re))]² + Te
    Te = Voo - De
    # estimates
    l1 = np.log(1 - np.sqrt((PTP[-1] - Te)/De))
    l2 = np.log(1 - np.sqrt((PTP[-2] - Te)/De))
    Re = (l2*RTP[-1] - l1*RTP[-2])/(l2 - l1)
    β = l1/(Re - RTP[-1])

    # fit Morse to outer turning points - adjusting β, Re, De; fixed Voo
    outer = len(PTP) // 2
    popt, pcov = curve_fit(lambda x, β, Re, De: Morse(x, β, Re, De, Voo), 
                           RTP[outer:], PTP[outer:], p0=[β, Re, De])
    err = np.sqrt(np.diag(pcov))

    subR = R > RTP[-1]
    P[subR] = Morse(R[subR], *popt, Voo) 

    if verbose:
        print('\nRKR: Outer limb  De[1-exp(β(Re-R))]² + Te')
        for par, val, er in zip(['β', 'Re', 'De', 'Voo'], popt, err):
            print(f'RKR:  {par:3s}  {val:12.2f}±{er:.2f}')
        print(f'RKR:  Te   {Voo - popt[-1]:12.2f} cm-1')


def outer_limb_LeRoy(R, P, RTP, PTP, De, Voo, verbose=True):
    # V(r) = V∞ - Cn/Rⁿ
    def LeRoy(x, Cn, n, Voo):
        return Voo - Cn/x**n
  
    n = int(np.log((Voo - PTP[-1])/(Voo - PTP[-2]))/np.log(RTP[-2]/RTP[-1]))
    Cn = (Voo - PTP[-1])*RTP[-1]**n

    # fit long-range potential curve to outer turning points - adjust Cn;
    #                                                        - fixed n, Voo
    outer = len(PTP)*2 // 3   # choose last 1/3 of points
    popt, pcov = curve_fit(lambda x, Cn: LeRoy(x, Cn, n, Voo), 
                           RTP[outer:], PTP[outer:], p0=[Cn])
    err = np.sqrt(np.diag(pcov))

    subR = R > RTP[-1]
    P[subR] = LeRoy(R[subR], Cn, n, Voo)

    if verbose:
        print('\nRKR: Outer limb  Voo - Cn/R^n')
        for par, val, er in zip(['Cn'], popt, err):
            print(f'RKR:  {par:3s}  {val:12.2f}±{er:.2f}')
        print(f'RKR: n = {n}, Voo = {Voo:8.2f} cm-1')
