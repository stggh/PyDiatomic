# -*- coding: utf-8 -*-
#####################################################
#   O2 X-state energy levels
#   G. Rouille, G. Millot, R. Saint-Loup, and H. Berger
#   J. Mol. Spectrosc. 154, 372-382 (1992).
#
#   Stephen.Gibson@anu.edu.au August 2010
#####################################################

import numpy as np


def _A(J, B, D, H, λ, λp, λpp, μ, μp, μpp, μppp):
    return (2*J + 1) * (B - 2*D*(J*J + J+1)\
           + H*(3*J**4 + 6*J**3 + 12*J*J + 10*J + 4)\
           - μ/2 - (μp/2)*(J*J + J+4)\
           - (μpp/2)*(J**4 + 2*J**3 + 13*J*J + 12*J + 8)\
           + (3*μppp/2)) - (λ+λp*(7*J*J + 7*J + 4)/3\
           + λpp*(11*J**4 + 22*J**3 + 39*J*J + 28*J + 8)/3)/(2*J + 1)


def _D(J, λ, λp, λpp):
    return (λ + λp*(J*J + J+1) +\
           λpp*(J**4 + 2*J**3 + 7*J*J + 6*J + 2))*2/(2*J + 1)


def F13split(J, B, D, H, λ, λp, λpp, μ, μp, μpp, μppp):
    a = _A(J, B, D, H, λ, λp, λpp, μ, μp, μpp, μppp)
    d = _D(J, λ, λp, λpp)
    return np.sqrt(a*a + J*(J+1)*d*d)


def F2level(J, B, D, H, λ, λp, λpp, μ, μp, μpp, μppp):
    x = J*(J+1)
    return B*x - D*x*x + H*x*x*x + 2*λ/3 + 2*λp*x/3 +\
           2*λpp*x*x/3 - μ - μp*x - μpp*x*x + μppp


def F13av(J, B, D, H, λ, λp, λpp, μ, μp, μpp, μppp):
    return B*(J*J + J+1) - D*(J**4 + 2*J**3 + 7*J*J + 6*J + 2) +\
           H*(J**6 + 3*J**5 + 18*J**4 + 31*J**3 + 33*J*J + 18*J + 4) -\
           λ/3 - λp*(J*J + J+4)/3 -\
           λpp*(J**4 + 2*J**3 + 13*J*J + 12*J + 8)/3 - 1.5*μ -\
           (μp/2)*(7*J*J + 7*J + 4) -\
           (μpp/2)*(11*J**4 + 22*J**3 + 39*J*J + 28*J + 8) +\
           (μppp/2)*(2*J*J + 2*J + 5)


def rouille(v, N, J):
    """ O2 X-state fine-structure energies Rouille et al. 
        J. Mol. Spectrosc. 154, 372-382 (1992).

    Parameters
    ----------
    v : int
        vibrational quantum number
    N : int
        rotational quantum number excluding spin
    J : int
        total rotational quantum number

    Note: J = N - F + 2, where F = Hund's case(b) fine-structure level
    
    Returns
    -------
    energy : float
        energy in cm-1 of level, refenced to non-existent level N=0, J=0
    """
 
    V = [0.0, 1556.36103, 3089.17317, 4598.76637, 6085.10234,
         7548.36948, 8988.7386]
    B = [1.43767953, 1.42186007, 1.40612292, 1.39042964,
         1.37477988, 1.35921426, 1.34366771]
    D = [4.83984515e-6, 4.83955885e-6, 4.83666796e-6, 4.83955311e-6,
         4.84017091e-6, 4.84084911e-6, 4.84513823e-6]
    H = [2.8e-12, 2.8e-12, 2.8e-12, 2.8e-12, 2.8e-12, 2.8e-12, 2.8e-12]
    λ = [1.984751322, 1.98957894, 1.99440656, 1.99923418, 1.99923418,
            1.99923418, 1.99923418]
    λp = [1.94521e-6, 2.10924e-6, 2.2733e-6, 2.4374e-6, 2.4374e-6,
               2.4374e-6, 2.4374e-6]
    λpp = [1.103e-11, 1.103e-11, 1.103e-11, 1.103e-11, 1.103e-11,
                1.103e-11, 1.103e-11]
    μ = [-8.425390e-3, -8.445771e-3, -8.466152e-3, -8.486533e-3,
          -8.486533e-3, -8.486533e-3, -8.486533e-3]
    μp = [-8.106e-9, -8.264e-9, -8.42e-9, -8.58e-9, -8.58e-9, -8.58e-9,
           -8.58e-9]
    μpp = [-4.7e-14, -4.7e-14, -4.7e-14, -4.7e-14, -4.7e-14, -4.7e-14,
            -4.7e-14]
    μppp = [0, 0, 0, 0, 0, 0, 0]

    f = N - J + 2

    if v > len(V):
        return -1
    if J < 0:
        return -1
    if N == J:
        return V[v] + F2level(N, B[v], D[v], H[v], λ[v], λp[v],
               λpp[v], μ[v], μp[v], μpp[v], μppp[v]) - 1.3316

    x = F13split(J, B[v], D[v], H[v], λ[v], λp[v], λpp[v], μ[v],
                 μp[v], μpp[v], μppp[v])

    F13 = F13av(J, B[v], D[v], H[v], λ[v], λp[v], λpp[v], μ[v],
                μp[v], μpp[v], μppp[v])

    if J == 0 and N == 1:
        return V[v] - 1.08574398
    else:
        return V[v] + F13 + x*(f-2) - 1.3316   # to match Amiot an Verges

if __name__ == '__main__':
    print('Calculation reproducing table 10 of Amiot+Verges'
          'Can. J. Phys. 59, 1391 (1981).')
    F = np.zeros(4)
    print(" N     F1         F2         F3")
    for v in range(1):
        for N in range(1, 32, 2):
            F[2] = rouille(v, N, N)
            for f in (1, 3):
                J = N - f + 2
                F[f] = rouille(v, N, J)
            print(f'{N:2d} {F[1]:10.5f} {F[2]:10.5f} {F[3]:10.5f}')
