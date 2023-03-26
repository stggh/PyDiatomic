import numpy as np
from scipy.optimize import least_squares
from scipy.special import genlaguerre, gamma
from scipy.linalg import svd


def Wei(r, Re, De, Voo, b, h=0., **ignore):
    """ Modified Wei potential curve
           Jai et al. J Chem Phys 137, 014101 (2012).

    Parameters
    ----------
    r : numpy 1d-array
        internuclear distance grid
    Re : float
        internuclear distance at equilibrium
    Voo : float
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
    #     raise SystemExit(f'Wei(r, Re, De, Te, b, h): error h={h}, '
    #                       'require |h| < 1')

    ebRe = np.exp(b*Re)
    ebr = np.exp(b*r)
    Te = Voo - De

    return De*(1 - ebRe*(1 - h)/(ebr - h*ebRe))**2 + Te


def Morse(r, Re=2, De=40000, Te=0, beta=1):
    # default parameters Morse oscillator 1. 10.1016/j.jms.2022.111621
    """Morse potential energy curve.

    Parameters
    ----------
    r : numpy 1d-array
        internuclear distance grid
    Re : float
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

    Voo = De + Te
    return Wei(r, Re, De, Voo, beta, h=0)


def Morse_wavefunction(r, Re=2, v=1, alpha=1, A=68.8885):
    # default parameters Morse oscillator 1. 10.1016/j.jms.2022.111621
    y = A*np.exp(-alpha*(r-Re))
    beta = A - 2*v - 1
    Nv = np.sqrt(alpha*beta*np.math.factorial(v)/gamma(A - v))
    wf = Nv*np.exp(-y/2)*y**(beta/2)*genlaguerre(v, beta)(y)
    return wf


def Julienne(r, Mx, Rx, Vx, Voo, **ignore):
    """Julienne (and Krauss) dissociative potential energy curve.

    math:
          V(r) = Vₓ exp[-(Mₓ/Vₓ)(r - Rₓ)] + V∞

    Eq. (45) J. Mol. Spect. 56, 270-308 (1975)

    Parameters
    ----------
    r : numpy 1d-array
        internuclear distance grid.
    Rx : float
        crossing point with a bound curve.
    Mx : float
        slope at crossing point (Rx).
    Vx : float
        energy at crossing point.
    Voo : float
        disociation limit energy.

    Returns
    -------
    potential_curve: numpy 1d-array
        potenial energy curve
    """

    return Vx*np.exp(-(Mx/Vx)*(r-Rx)) + Voo


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


def Wei_fit(r, V, Re=None, De=None, Voo=None, b=1., h=0.1,
            adjust=['Re', 'De', 'b', 'h'], verbose=False):

    def residual(pars, adjust, paramdict, r, V):
        for i, x in enumerate(adjust):
            paramdict[x] = pars[i]

        return Wei(r, *list(paramdict.values())) - V

    if Re is None:
        Re = r[V.argmin()]
    if Voo is None:
        Voo = V[-1]
    if De is None:
        De = Voo - V.min()

    paramdict = {'Re':Re, 'De':De, 'Voo':Voo, 'b':b, 'h':h}
    unit = {'Re':'Å', 'De':'cm⁻¹', 'Voo':'cm⁻¹', 'b':'', 'h':''}
    lower_bound = {'Re':0.1, 'De':0, 'Voo':-100, 'b':0.1, 'h':-1}
    upper_bound = {'Re':5, 'De':1e5, 'Voo':1e5, 'b':5, 'h':1}

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

def Julienne_fit(r, V, Mx=None, Rx=None, Vx=None, Voo=None,
                 adjust=['Mx', 'Rx', 'Vx'], verbose=False):
    def residual(pars, adjust, paramdict, r, V):
        for i, x in enumerate(adjust):
            paramdict[x] = pars[i]

        return Julienne(r, *list(paramdict.values())) - V

    if Voo is None:
        Voo = V[-1]
    if Vx is None:
        Vx = 1.1*voo
    if Rx is None:
        Rx = r[np.abs(V - Vx).argmin()]
    if Mx is None:
        Mx = 1e4

    paramdict = {'Mx':Mx, 'Rx':Rx, 'Vx':Vx, 'Voo':Voo}
    unit = {'Mx':'cm⁻¹/Å', 'Rx':'Å', 'Vx':'cm⁻¹', 'Voo':'cm⁻¹'}
    lower_bound = {'Mx':1e2, 'Rx':0.5, 'Vx':-100, 'Voo':-100}
    upper_bound = {'Mx':1e5, 'Rx':5, 'Vx':1e6, 'Voo':1e5}

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
