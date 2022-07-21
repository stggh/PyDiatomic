import numpy as np
from scipy.optimize import least_squares
from scipy.special import genlaguerre, gamma
from scipy.linalg import svd


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


def Morse(r, re=2, De=40000, Te=0, beta=1):
    # default parameters Morse oscillator 1. 10.1016/j.jms.2022.111621
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


def Morse_wavefunction(r, re=2, v=1, alpha=1, A=68.8885):
    # default parameters Morse oscillator 1. 10.1016/j.jms.2022.111621
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


def fiterrors(result):
    ''' from:
        https://stackoverflow.com/questions/42388139/how-to-compute-standard-deviation-errors-with-scipy-optimize-least-squares

    '''
    U, s, Vh = svd(result.jac, full_matrices=False)
    tol = np.finfo(float).eps*s[0]*max(result.jac.shape)
    w = s > tol
    cov = (Vh[w].T/s[w]**2) @ Vh[w]  # robust covariance matrix
    chi2dof = np.sum(result.fun**2)/(result.fun.size - result.x.size)
    cov *= chi2dof
    return np.sqrt(np.diag(cov))


def Wei_fit(r, V, re=None, De=None, Te=None, b=1., h=0.1, verbose=False):
    def residual(pars, De, r, V):
        re, Te, b, h = pars
        return Wei(r, re, De, Te, b, h) - V

    if re is None:
        re = r[V.argmin()]
    if Te is None:
        Te = V.min()
    if De is None:
        De = V[-1] - Te

    pars = [re, Te, b, h]
    result = least_squares(residual, pars, args=(De, r, V),
                           bounds=([0.1, -100., 0.1, -0.1],
                                   [5., 1.e5, 5., 0.1]))
    result.stderr = fiterrors(result)
    if verbose:
        print('Wei_fit:')
        print(f'  re = {result.x[0]:5.3f}±{result.stderr[0]:.3f} Å')
        print(f'  Te = {result.x[1]:7.3f}±{result.stderr[1]:.3f} cm⁻¹')
        print(f'  De = {De:7.3f} (fixed) cm⁻¹')
        print(f'  b = {result.x[2]:5.3f}±{result.stderr[2]:.3f}')
        print(f'  h = {result.x[3]:5.3f}±{result.stderr[3]:.3f}')
    return result

def Julienne_fit(r, V, mx=None, rx=None, vx=None, voo=None):
    def residual(pars, r, V):
        mx, rx, vx, voo = pars
        return Julienne(r, mx, rx, vx, voo) - V

    if voo is None:
        voo = V[-1]
    if vx is None:
        vx = 1.1*voo
    if rx is None:
        rx = r[np.abs(V - vx).argmin()]
    if mx is None:
        mx = 1e4

    pars = [mx, rx, vx, voo]
    result = least_squares(residual, pars, args=(r, V),
                           bounds=([1, 0.1, 0, 0], [1e8, 5, 1e5, 1e5])) 
    result.stderr = fiterrors(result)
    return result
