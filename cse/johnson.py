# -*- coding: utf-8 -*-
import numpy as np
import scipy.constants as const

from scipy.optimize import least_squares
from scipy.integrate.quadrature import simps
from scipy.special import spherical_jn, spherical_yn
from scipy.signal import find_peaks

##############################################################################
#  PyDiatomic - solve the coupled-channel time-independent Schrödinger equation
#               using recipe of B.R. Johnson J Chem Phys 69, 4678 (1977).
#
#  Stephen.Gibson@anu.edu.au
#  January 2016
##############################################################################


# ===== Johnson === Renormalized Numerov method ============

def WImat(energy, rot, V, R, μ, AM):
    """ Interaction matrix.

    Parameters
    ----------
    energy : float
        energy in eV or estimate (for bound states) at which to solve the TISE
    rot : int
        rotational quantum number J
    V : numpy 3d array - shape oo, n, n = radial grid size, number of channels
        potential energy curve matrix, diagonals are the diabatic potential
        energy curves, off-diagonals the coupling
    R : numpy 1d array
        the internuclear distance grid for the potential curve matrix,
        size = oo
    μ : float
        reduced mass in kg
    AM : 1d numpy array of tuples
        (Ω, S, Λ, Σ) for each electronic state

    Returns
    -------
    WI : numpy 3d array - shape oo, n, n
       inverse of the interation matrix W

    """

    dR2 = (R[1] - R[0])**2

    # 2μ/hbar^2 x \DeltaR^2/12 x e
    factor = μ*1.0e-20*dR2*const.e/const.hbar/const.hbar/6

    # hbar^2/2μ x e x 10^20
    centrifugal_factor = (const.hbar*1.0e20/μ/2/const.e)*const.hbar

    oo, n, m = V.shape
    I = np.identity(n)
    barrier = V.copy().T

    # Fix me! - needs a tidy-up
    if rot:
        Jp1 = rot*(rot+1)
        for j in range(n):
            Ω, S, Λ, Σ = AM[j]
            am = Ω**2 - S*(S+1) + Σ**2
            # diagonal - add centrifugal barrier to potential curve
            if Jp1 > am:
                barrier[j, j, :] += (Jp1 - am)*centrifugal_factor/R[:]**2

            for k in range(j+1, n):
                Ωk, Sk, Σk, Λk = AM[k]
                # off-diagonal
                if Ω != Ωk:
                    # L-uncoupling, homogeneous coupling already set
                    if Jp1 > Ω*Ωk:
                        barrier[j, k, :] = barrier[k, j, :] = 8064.541*\
                              V[:, j, k]*np.sqrt(Jp1 - Ω*Ωk)*\
                              centrifugal_factor/R[:]**2
                                   
    barrier = barrier.T

    # generate W^-1
    WI = np.zeros_like(V)
    WI[:] = np.linalg.inv(I + (energy*I - barrier[:])*factor)

    return WI


def RImat(WI, mx):
    """ R matrix

    Parameters
    ----------
    WI : numpy 3d array
        array of inverted interaction arrays, as returned from WImat

    mx: int
        matching point in inward and outward solutions (bound states only)

    Returns
    -------
    RI : numpy 3d array
        R matrix of the Johnson method

    """

    oo, n, m = WI.shape
    I = np.identity(n)
    RI = np.zeros_like(WI)

    U = 12*WI-I*10
    for i in range(1, mx+1):
        RI[i] = np.linalg.inv(U[i]-RI[i-1])

    for i in range(oo-2, mx, -1):
        RI[i] = np.linalg.inv(U[i]-RI[i+1])

    return RI


def fmat(j, RI, WI,  mx):
    """ f-matrix of the Johnson method.

    Parameters
    ----------
    j : int
        open channel number
    RI : numpy 3d array
        inverted R-matrix as returned from RImat
    WI : numpy 3d array
        inverted interaction matrix as returned from WImat
    mx : int
        matching point for inward and outward solutions

    Returns
    -------
    f : numpy 2d array
        Johnson f-matrix

    """
    oo, n, m = WI.shape
    f = np.zeros((oo, n))

    if n == 1 or mx > oo-20:
        # single PEC or unbound
        f[mx] = np.linalg.inv(WI[mx])[j]
    else:
        # (R_m - R^-1_m+1).f(R) = 0
        U, s, Vh = np.linalg.svd(np.linalg.inv(RI[mx-1])-RI[mx])
        # for i, x in enumerate(s):
        #     if x > 0:
        #         break  # any diagonal !=0 yields a solution

        # Fix me! Gives correct inward solution wavefunction phase
        #  for O2X fine-structure
        U = U.T
        f[mx] = U[1] if U[1, 0] < 0 else U[-1]

    for i in range(mx-1, -1, -1):
        f[i] = f[i+1] @ RI[i]

    for i in range(mx+1, oo):
        f[i] = f[i-1] @ RI[i]
    return f


def wavefunction(WI, j, f):
    """ evaluate wavefunctions from f-matrix array.

    Parameters
    ----------
    WI : numpy 3d array
        inverted interaction matrix, as returned from WImat
    j : int
        open channel number
    f : numpy 2d array
        f-matrix as returned from fmat

    Returns
    -------
    wf : numpy 3d array
        oo x n x nopen array of wavefunctions

    """
    oo, n = f.shape
    wf = np.zeros_like(f)

    for i in range(oo):
        wf[i] = f[i] @ WI[i]

    return np.transpose(wf)

# ==== end of Johnson stuff ====================

def node_positions(WI, mn, mx):
    """ find inner and outer solution nodes, before the # outer turning point.

    """

    oo, n, m = WI.shape
    RIoutwards = RImat(WI, oo-1)[mn:mx]
    RIinwards = RImat(WI, 1)[mn:mx]

    if n == 1:  # det(RI) works better than det(R)
        detIR_out = RIoutwards[:, 0, 0]
        detIR_in = RIinwards[:, 0, 0]
    else:
        detIR_out = np.linalg.det(RIoutwards)
        detIR_in = np.linalg.det(RIinwards)

    # determine the node positions
    inner, _ = find_peaks(detIR_in)
    outer, _ = find_peaks(detIR_out)

    return inner, outer


def matching_point(en, rot, V, R, μ, AM):
    """ estimate matching point for inward and outward solutions position
    based on the determinant of the R-matrix.

    Parameters
    ----------
    en : float
        potential energy of the solution
    rot : int
        rotational quantum number J
    V : numpy 3d array
        potential curve and couplings matrix
    R : numpy 1d array
        internuclear distance grid
    μ : float
        reduced mass in kg
    AM : 1d numpuy array of tuples
        (Ω, S, Λ, Σ) for each electronic state

    Returns
    -------
    mx : int
        matching point grid index
    """

    oo, n, m = V.shape

    # index for potential energy curve with lowest dissociation energy
    jm = np.array([V[-1, j, j] for j in range(n)]).argmin()
    Vm = V[-1, jm, jm]  # dissociation energy

    if en > Vm:  #  at least one open channel
        return oo-1
    else:  # all channels closed, determine matching point
        jRe = V[:, jm, jm].argmin()  # potential energy index of minimum
        # inner and outer crossing point indices for energy en
        # //2, *2 factor broadens range
        mn = np.abs(V[:jRe, jm, jm] - en).argmin()//2  # inner 
        mx = np.abs(V[jRe:, jm, jm] - en).argmin()*2 + jRe  # outer
        mx = min(oo, mx)

        WI = WImat(en, rot, V, R, μ, AM)
        inner, outer = node_positions(WI, mn, mx)

        # Johnson uses two energies that bracket the eigenvalue,
        # here take outermost region, as not obvious how to bracket eigenvalue
        vib = len(outer)
        if len(outer) > 0:
            mx = (outer[-1] + inner[-1])//2 + mn

    return mx


def eigen(energy, rot, mx, V, R, μ, AM):
    """ determine eigen energy solution based.

    Parameters
    ----------
    energy : float
        energy (eV) of the attempted solution
    rot : int
        rotational quantum number
    mx : int
        matching point index, for inward and outward solutions
    V : numpy 3d array
        potential energy curve and coupling matrix
    R : numpy 1d array
        internuclear distance grid
    μ : float
        reduced mass in kg

    Returns
    -------
    determinant : float
        | R_mx - R^-1_mx+1 | 

    """

    WI = WImat(energy, rot, V, R, μ, AM)
    RI = RImat(WI, mx)

    # | R_mx - R^-1_mx+1 |     x1000 scalinhg helps leeastsquares
    return np.linalg.det(np.linalg.inv(RI[mx])-RI[mx+1])*1000


def normalize(wf, R):
    """ normalize a bound state wavefunction

    Parameters
    ----------
    wf : numpy 3d array
        wavefunction array

    R : numpy 1d array
        internuclear distance grid

    Returns
    -------
    wf: numpy 3d array
        normalized wavefunction array

    """
    oo, n, nopen = wf.shape
    norm = 0.0
    for j in range(n):
        norm += simps(wf[:, j, 0]**2, R)

    return wf/np.sqrt(norm)


def amplitude(wf, R, edash, μ):
    # Mies    F ~ JA + NB       J ~ sin(kR)/kR
    # normalization sqrt(2 μ/π hbar^2) = zz
    π = const.pi
    zz = np.sqrt(2*μ/π)/const.hbar

    oo, n, nopen = wf.shape

    # two asymptotic points on wavefunction wf[:, j]
    i1 = oo-5
    i2 = i1-1
    x1 = R[i1]*1.0e-10
    x2 = R[i2]*1.0e-10

    A = np.zeros((nopen, nopen))
    B = np.zeros((nopen, nopen))
    oc = 0
    for j in range(n):
        if edash[j] < 0:
            continue
        # open channel
        ke = np.sqrt(2*μ*edash[j]*const.e)/const.hbar
        rtk = np.sqrt(ke)
        kex1 = ke*x1
        kex2 = ke*x2

        j1 = spherical_jn(0, kex1)*x1*rtk*zz
        y1 = spherical_yn(0, kex1)*x1*rtk*zz

        j2 = spherical_jn(0, kex2)*x2*rtk*zz
        y2 = spherical_yn(0, kex2)*x2*rtk*zz

        det = j1*y2 - j2*y1

        for k in range(nopen):
            A[oc, k] = (y2*wf[i1, j, k] - y1*wf[i2, j, k])/det
            B[oc, k] = (j1*wf[i2, j, k] - j2*wf[i1, j, k])/det

        oc += 1

    AI = np.linalg.inv(A)
    K = B @ AI

    return K, AI, B


def solveCSE(Cse, en):

    rot = Cse.rot
    μ = Cse.μ
    R = Cse.R
    AM = Cse.AM
    VT = Cse.VT
    n, m, oo = VT.shape

    V = np.transpose(VT)

    # find channels that are open, as defined by E' > 0
    edash = en - np.diag(VT[:, :, -1])
    openchann = edash > 0
    nopen = edash[openchann].size

    mx = matching_point(en, rot, V, R, μ, AM)
    Cse.mx = mx

    if mx < oo-5:
        out = least_squares(eigen, (en, ), args=(rot, mx, V, R, μ, AM))
        en = float(out.x[0])

    # solve CSE according to Johnson renormalized Numerov method
    WI = WImat(en, rot, V, R, μ, AM)
    RI = RImat(WI, mx)
    wf = []
    if nopen > 0:
        oc = 0
        for j, ed in enumerate(edash):
            if ed > 0:
                f = fmat(j, RI, WI, mx)
                wf.append(wavefunction(WI, oc, f))
                oc += 1
    else:
        f = fmat(0, RI, WI, mx)
        wf.append(wavefunction(WI, nopen, f))

    wf = np.array(wf)
    wf = np.transpose(wf)

    if nopen == 0:
        wf = normalize(wf, R)
    else:
        K, AI, B = amplitude(wf, R, edash, μ)   # shape = nopen x nopen

        # K = BA-1 = U tan xi UT
        eig, U = np.linalg.eig(K)

        # form A^-1 U cos xi exp(i xi) UT
        I = np.identity(nopen, dtype=complex)
        xi = np.arctan(eig)*I
        cosxi = np.cos(xi)*I
        expxi = np.exp((0+1j)*xi)*I

        expxiUT = expxi @ np.transpose(U)
        cosxiexpxiUT = cosxi @ expxiUT

        UcosxiexpxiUT = U @ cosxiexpxiUT
        Norm = AI @ UcosxiexpxiUT    # == (cu, su) complex

        # complex wavefunction array  oo x n x nopen
        wf = wf @ Norm

    return wf, en, openchann
