import numpy as np
from scipy.optimize import least_squares
from scipy.special import genlaguerre, gamma


def Wei(r, re, De, Te, b, h=0.):
    """ Modified Wei potential curve
           Jai et al. J Chem Phys 137, 014101 (2012).

    Parameters
    ----------
    r : numpy 1d-array
        internuclear distance grid
    re : float
        internuclear distance at equilibrium
    De : float
        Dissociation energy
    Te : float
        equilibrium energy (potential minimum)
    b : float
        anharmonicity parameter
    h : float
        |h| < 1, h=0 gives a Morse potential curve

    Returns
    -------
    potential_curve : numpy 1d-array
        potential energy curve

    """

    # if abs(h) > 1:
    #     raise SystemExit(f'Wei(r, re, De, Te, b, h): error h={h}, '
    #                       'require |h| < 1')

    ebre = np.exp(b*re)
    ebr = np.exp(b*r)

    return De*(1 - ebre*(1 - h)/(ebr - h*ebre))**2 + Te


def Wei_fit(r, V, re, De, Te=1., b=1., h=0.):
    def residual(pars, r, V):
        re, De, Te, b, h = pars
        return Wei(r, re, De, Te, b, h) - V

    pars = [re, De, Te, b, h]
    result = least_squares(residual, pars, args=(r, V),
                           bounds=([0.1, 100., -1000., 0.1, -0.1],
                                   [10., 1.e8, 1.e8, 5., 0.1]))
    return result


def Morse(r, re, De, Te, beta):
    """Morse potential energy curve.

    Parameters
    ----------
    r : numpy 1d-array
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

    return Wei(r, re, De, Te, beta, h=0)


def Morse_wavefunction(r, re, v, alpha, A):
    y = A*np.exp(-alpha*(r-re))
    beta = A - 2*v - 1
    Nv = np.sqrt(alpha*beta*np.math.factorial(v)/gamma(A - v))
    wf = Nv*np.exp(-y/2)*y**(beta/2)*genlaguerre(v, beta)(y)
    return wf


def Julienne(r, mx, rx, vx, voo):
    """Julienne (and Krauss) dissociative potential energy curve.

    math:
          v(r) = v_x \exp[-(m_x/v_x)(r - r_x)] + v_\infy

    Eq. (45) J. Mol. Spect. 56, 270-308 (1975)

    Parameters
    ----------
    r : numpy 1d-array
        internuclear distance grid.
    rx : float
        crossing point with a bound curve.
    mx : float
        slope at crossing point (rx).
    vx : float
        energy at crossing point.
    voo : float
        disociation limit energy.

    Returns
    -------
    potential_curve: numpy 1d-array
        potenial energy curve
    """

    return vx*np.exp(-(mx/vx)*(r-rx)) + voo


def Julienne_fit(r, V, mx, rx, vx, voo):
    def residual(pars, r, V):
        mx, rx, vx, voo = pars
        return Julienne(r, mx, rx, vx, voo) - V

    pars = [mx, rx, vx, voo]
    result = least_squares(residual, pars, args=(r, V))
    return result
