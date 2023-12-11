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
    """ Rydberg-Klein-Rees potential energy curve.

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

    Rmin, Rmax, E, vib = turning_points(μ, vv, Gv, Bv, dv, verbose=verbose)

    PEC, vib, RTP, PTP = formPEC(Rgrid, vib, Rmin, Rmax, E, De, Voo, limb,
                                 ineV, verbose=verbose)

    return Rgrid, PEC, vib, RTP, PTP


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
    vib = np.arange(-1/2, vv[-1]+dv/2, dv)
    # offset the point at v=-1/2, the bottom of the well
    vib[0] = -1/2 + 0.0001

    # compute turning points using RKR method
    E = G(vib, gsp)
    ff = fg_integral(vib, gsp, bsp, lambda x, y: 1)
    gg = fg_integral(vib, gsp, bsp, B)
    fg = np.sqrt(ff/gg + ff**2)

    Rmin = (fg - ff)*DD  # turning points
    Rmax = (fg + ff)*DD

    if verbose:
        vint = np.mod(vib, 1) > 0.99  # integer levels
        Re = (Rmin[0] + Rmax[0])/2
        
        print('RKR:  v    Rmin(Å)   Rmax(Å)     E(cm⁻¹)')
        print(f'    -1/2       {Re:7.3f}         {E[0]:9.2f}')
        for v, rm, rx, e in zip(vib[vint], Rmin[vint], Rmax[vint], E[vint]):
            print(f'    {round(v):3d}   {rm:7.3f}   {rx:7.3f}    {e:9.2f}')

        print(f'\nRKR: Rₑ = {Re:5.3f} Å, Tₑ = {E[0]:,.2f} cm⁻¹')


    return Rmin, Rmax, E, vib


def formPEC(R, vib, Rmin, Rmax, E, De, Voo, limb, ineV=False, verbose=True):
    evcm = const.e/(const.c*const.h*100)  # converts cm-1 to eV

    # combine Rmin with Rmax to form PEC
    Re = (Rmin[0] + Rmax[0])/2

    vib = np.append(vib[-1::-1], vib)  # fractional vibrational "quantum number"
    RTP = np.append(Rmin[::-1], Rmax, 0)  # radial positions of turning-points
    PTP = np.append(E[::-1], E, 0)  # potential energy at turning-points

    # truncate inner-limb if not increasing, i.e. turns-in
    subR = np.diff(RTP, prepend=2*RTP[0]-RTP[1]) > 0
    vib = vib[subR]
    RTP = RTP[subR]
    PTP = PTP[subR]

    # Interpolate RKR turning points to internuclear grid, R
    psp = splrep(RTP, PTP, s=0)
    PEC = splev(R, psp, der=0)

    # minimum is Te
    bound = np.logical_and(R > RTP[0], R < RTP[-1])
    Te = PEC[bound].min()

    # extend/extrapolate using analytical function
    inner_limb_Morse(R, PEC, RTP, PTP, Re, De, Voo, verbose=verbose)
    if limb == 'L':
        outer_limb_LeRoy(R, PEC, RTP, PTP, De, Voo, verbose=verbose)
    else:
        outer_limb_Morse(R, PEC, RTP, PTP, Re, De, Voo, verbose=verbose)

    if ineV:  # convert to eV
        PTP /= evcm
        PEC /= evcm

    return PEC, vib, RTP, PTP

def Morse(x, β, Re, De, Voo):
    # Dₑ[1 - exp(-β(x-Rₑ))]² + Tₑ
    Te = Voo - De
    return De*(1 - np.exp(-β*(x-Re)))**2 + Te

# analytical functions
def inner_limb_Morse(R, P, RTP, PTP, Re, De, Voo, verbose=True):
    # Dₑ[1 - exp(-β(R-Rₑ))]² + Tₑ
    Te = Voo - De

    # evaluate β based on inner-most turning point
    ln0 = np.log(1 + np.sqrt((PTP[0] - Te)/De))
    β = ln0/(Re - RTP[0])

    # fit Morse to inner turning points - adjusting β, Re, De; fixed Voo
    # inner = len(PTP) // 2
    inner = RTP < Re*0.95

    popt, pcov = curve_fit(lambda x, β, Re, De: Morse(x, β, Re, De, Voo),
                           RTP[inner], PTP[inner], p0=[β, Re, De])
    err = np.sqrt(np.diag(pcov))

    subR = R < RTP[0]
    P[subR] = Morse(R[subR], *popt, Voo)

    if verbose:
        print('\nRKR: Inner limb  Morse: Dₑ[1 - exp(-β(R-Rₑ))]² + Tₑ')
        for par, val, er in zip(['β', 'Rₑ', 'Dₑ', 'V∞'], popt, err):
            print(f'RKR:  {par:3s}  {val:12.2f} ± {er:.2f}')
        print(f'RKR:  Tₑ   {Voo - popt[-1]:,.2f} cm⁻¹')


def outer_limb_Morse(R, P, RTP, PTP, Re, De, Voo, verbose=True):
    # Dₑ[1 - exp(-β(R-Rₑ))]² + Tₑ
    Te = Voo - De
    # estimates
    ix = -1
    l1 = np.log(1 - np.sqrt((PTP[ix] - Te)/De))
    l2 = np.log(1 - np.sqrt((PTP[ix-1] - Te)/De))

    ReM = (l2*RTP[ix] - l1*RTP[ix-1])/(l2 - l1)
    β = l1/(ReM - RTP[ix])

    # fit Morse to outer turning points - adjusting β, Re, De; fixed Voo
    outer = len(PTP) // 6 
    subR = R > RTP[ix]

    try:
        popt, pcov = curve_fit(lambda x, β, Re, De: Morse(x, β, Re, De, Voo), 
                               RTP[-outer:], PTP[-outer:], p0=[β, ReM, De])
        err = np.sqrt(np.diag(pcov))

        # Morse function extrapolation + offset to correct R[-1] != ∞
        P[subR] = Morse(R[subR], *popt, Voo) + Morse(R[ix], *popt, Voo)

        if verbose:
            print('\nRKR: Outer limb Morse: Dₑ[1 - exp(-β(R-Rₑ))]² + Tₑ')
            for par, val, er in zip(['β', 'Rₑ', 'Dₑ', 'V∞'], popt, err):
                print(f'RKR:  {par:3s}  {val:12.2f} ± {er:.2f}')
            print(f'RKR:  Tₑ   {Voo - popt[ix]:,.2f} cm⁻¹')

    except:
        P[subR] = Morse(R[subR], β, ReM, De, Voo)
        print('\nRKR: Outer limb Morse: Dₑ[1 - exp(-β(R-Rₑ))]² + Tₑ')
        print(f'l1={l1:5.3f}, l2={l2:5.3f}')
        print(f'Re={ReM:5.3f}, β = {β:5.3g}, Te = {Te:,.2f}, '
              f'De = {De:,.2f}, Voo = {Voo:,.2f}')


def outer_limb_LeRoy(R, P, RTP, PTP, De, Voo, verbose=True):
    # V∞ - Cₙ/Rⁿ
    def LeRoy(x, Cn, n, Voo):
        return Voo - Cn/x**n
  
    # estimates based on last turning point
    n = np.log((Voo - PTP[-1])/(Voo - PTP[-2]))/np.log(RTP[-2]/RTP[-1])
    if n < 1.1:
        n = 1.1
    Cn = (Voo - PTP[-1])*RTP[-1]**n

    # fit long-range potential curve to outer turning points - adjust Cn;
    #                                                        - fixed n, Voo
    outer = len(PTP) // 6

    popt, pcov = curve_fit(lambda x, Cn, n: LeRoy(x, Cn, n, Voo), 
                           RTP[-outer:], PTP[-outer:], p0=[Cn, n],
                           bounds=((1, 1.1), (np.inf, 20)))
    err = np.sqrt(np.diag(pcov))

    subR = R > RTP[-1]
    # LeRoy function extrapolation + offset to correct R[-1] != ∞
    P[subR] = LeRoy(R[subR], Cn, n, Voo) # + Cn/R[-1]**n

    if verbose:
        print('\nRKR: Outer limb  LeRoy: V∞ - Cₙ/Rⁿ')
        for par, val, er in zip(['Cₙ'], popt, err):
            print(f'RKR:  {par:3s}  {val:12.2f} ± {er:.2f}')
        print(f'RKR: n = {n}, V∞ = {Voo:,.2f} cm⁻¹')
