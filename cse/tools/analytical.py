import numpy as np
from scipy.optimize import least_squares
from scipy.special import genlaguerre, gamma
from scipy.linalg import svd


def Wei(r, re, De, voo, b, h=0.):
    """ Modified Wei potential curve
           Jai et al. J Chem Phys 137, 014101 (2012).

    Parameters
    ----------
    r : numpy 1d-array
        internuclear distance grid
    re : float
        internuclear distance at equilibrium
    voo : float
        dissociation energy
    De : float
        potential well depth
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
    Te = voo - De

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

    voo = De + Te
    return Wei(r, re, De, voo, beta, h=0)


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


def Wei_fit(r, V, re=None, De=None, voo=None, b=1., h=0.1, 
            adjust=['re', 'De', 'b', 'h'], verbose=False):

    def residual(pars, adjust, paramdict, r, V):
        for i, x in enumerate(adjust):
            paramdict[x] = pars[i]

        return Wei(r, *list(paramdict.values())) - V

    if re is None:
        re = r[V.argmin()]
    if voo is None:
        voo = V[-1]
    if De is None:
        De = voo - V.min()

    paramdict = {'re':re, 'De':De, 'voo':voo, 'b':b, 'h':h}
    unit = {'re':'Å', 'De':'cm⁻¹', 'voo':'cm⁻¹', 'b':'', 'h':''}
    lower_bound = {'re':0.1, 'De':0, 'voo':-100, 'b':0.1, 'h':-1}
    upper_bound = {'re':5, 'De':1e5, 'voo':1e5, 'b':5, 'h':1}

    pars = [paramdict[x] for x in adjust]
    lb = [lower_bound[x] for x in adjust]
    ub = [upper_bound[x] for x in adjust]

    result = least_squares(residual, pars, args=(adjust, paramdict, r, V),
                           bounds=(lb, ub))

    result.stderr = fiterrors(result)
    for i, x in enumerate(adjust):
        paramdict[x] = result.x[i]

        i = 0
        fitstr = ''
        for k, v in paramdict.items():
            fitstr += f'Wei_fit:  {k:>5s} = {v:8.3f}'
            if k in adjust:
                fitstr += f' ± {result.stderr[i]:.3f} {unit[k]}\n'
                i += 1
            else:
                fitstr += f' {unit[k]} (fixed)\n'
        result.fitstr = fitstr

    if verbose:
        print(fitstr)

    result.paramdict = paramdict
    return result

def Julienne_fit(r, V, mx=None, rx=None, vx=None, voo=None,
                 adjust=['mx', 'rx', 'vx'], verbose=False):
    def residual(pars, adjust, paramdict, r, V):
        for i, x in enumerate(adjust):
            paramdict[x] = pars[i]

        return Julienne(r, *list(paramdict.values())) - V

    if voo is None:
        voo = V[-1]
    if vx is None:
        vx = 1.1*voo
    if rx is None:
        rx = r[np.abs(V - vx).argmin()]
    if mx is None:
        mx = 1e4

    paramdict = {'mx':mx, 'rx':rx, 'vx':vx, 'voo':voo}
    unit = {'mx':'cm⁻¹/Å', 'rx':'Å', 'vx':'cm⁻¹', 'voo':'cm⁻¹'}
    lower_bound = {'mx':1e2, 'rx':0.5, 'vx':-100, 'voo':-100}
    upper_bound = {'mx':1e5, 'rx':5, 'vx':1e5, 'voo':1e5}

    pars = [paramdict[x] for x in adjust]
    lb = [lower_bound[x] for x in adjust]
    ub = [upper_bound[x] for x in adjust]

    result = least_squares(residual, pars, args=(adjust, paramdict, r, V),
                           bounds=(lb, ub))

    result.stderr = fiterrors(result)
    for i, x in enumerate(adjust):
        paramdict[x] = result.x[i]

        i = 0
        fitstr = ''
        for k, v in paramdict.items():
            fitstr += f'Julienne_fit:  {k:>5s} = {v:8.3f}'
            if k in adjust:
                fitstr += f' ± {result.stderr[i]:.3f} {unit[k]}\n'
                i += 1
            else:
                fitstr += f' {unit[k]} (fixed)\n'
        result.fitstr = fitstr

    if verbose:
        print(fitstr)

    result.paramdict = paramdict
    return result
