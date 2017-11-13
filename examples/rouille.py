#########################################################################
#
# exact O2 X-state fine-structure levels
#    G. Rouille, G. Millot, R. Saint-Loup, and H. Berger
#    J. Mol. Spectrosc. 154, 372-382 (1992)
#
# Stephen.Gibson@anu.edu.au - 14 November 2017
#########################################################################

import numpy as np


def A_(J, B, D, H, lamb, lambdap, lambdapp, mu, mup, mupp, muppp):
    return (2*J + 1)*(B - 2*D*(J*J + J+1) +
           H*(3*J**4 + 6*J**3 + 12*J*J + 10*J + 4) - mu/2 -
           mup*(J*J + J + 4)/2 - mupp*(J**4 + 2*J**3 + 13*J*J + 12*J + 8)/2 +
           3*muppp/2) - (lamb + lambdap*(7*J*J + 7*J + 4)/3 +
           lambdapp*(11*J**4 + 22*J**3 + 39*J*J + 28*J + 8)/3)/(2*J + 1)


def D_(J, lamb, lambdap, lambdapp):
    return (lamb + lambdap*(J*J + J + 1) +
            lambdapp*(J**4 + 2*J**3 + 7*J*J + 6*J + 2))*2/(2*J + 1)


def F13split(J, B, D, H, lamb, lambdap, lambdapp, mu, mup, mupp, muppp):
    a = A_(J, B, D, H, lamb, lambdap, lambdapp, mu, mup, mupp, muppp)
    d = D_(J, lamb, lambdap, lambdapp)

    return np.sqrt(a*a + J*(J+1)*d*d)


def F2level(J, B, D, H, lamb, lambdap, lambdapp, mu, mup, mupp, muppp):
    x = J*(J+1)

    return B*x - D*x*x + H*x*x*x + 2*lamb/3 + 2*lambdap*x/3 +\
           2*lambdapp*x*x/3 - mu - mup*x - mupp*x*x + muppp


def F13av(J, B, D, H, lamb, lambdap, lambdapp, mu, mup, mupp, muppp):
    return B*(J*J + J+1) - D*(J**4 + 2*J**3 + 7*J*J + 6*J + 2) +\
           H*(J**6 + 3*J**5 + 18*J**4 + 31*J**3 + 33*J*J + 18*J + 4) -\
           lamb/3 - lambdap*(J*J + J+4)/3 -\
           lambdapp*(J**4 + 2*J**3 + 13*J*J + 12*J + 8)/3 - 1.5*mu -\
           mup*(7*J*J + 7*J + 4)/2 -\
           mupp*(11*J**4 + 22*J**3 + 39*J*J + 28*J + 8)/2 +\
           muppp*(2*J*J + 2*J + 5)/2


def rouille_(v, N, J):
    offset = 1.3316
    V = np.array([0, 1556.38991, 1532.86724, 1509.5275])
    B = np.array([1.437676476, 1.42186454, 1.4061199, 1.390425])
    D = np.array([4.84256e-6, 4.8418e-6, 4.841e-6, 4.8402e-6])
    H = np.array([2.8e-12, 2.8e-12, 2.8e-12, 2.8e-12])
    lamb = np.array([1.984751322, 1.98957894, 1.99440656, 1.99923418])
    lambdap = np.array([1.94521e-6, 2.10924e-6, 2.2733e-6, 2.4374e-6])
    lambdapp = np.array([1.103e-11, 1.103e-11, 1.103e-11, 1.103e-11])
    mu = np.array([-8.425390e-3, -8.445771e-3, -8.466152e-3, -8.486533e-3])
    mup = np.array([-8.106e-9, -8.264e-9, -8.42e-9, -8.58e-9])
    mupp = np.array([-4.7e-14, -4.7e-14, -4.7e-14, -4.7e-14])
    muppp = np.zeros(4)

    f = N - J + 2

    V[2] += V[1]
    V[3] += V[2]

    if v > 3 or J < 0:
        return -1

    if N == J:
        return V[v] + F2level(N, B[v], D[v], H[v], lamb[v], lambdap[v],
                              lambdapp[v], mu[v], mup[v], mupp[v], muppp[v])\
                    - offset

    x = F13split(J, B[v], D[v], H[v], lamb[v], lambdap[v], lambdapp[v],
                 mu[v], mup[v], mupp[v], muppp[v])
    F13 = F13av(J, B[v], D[v], H[v], lamb[v], lambdap[v], lambdapp[v], mu[v],
                mup[v], mupp[v], muppp[v])
    if J == 0 and N == 1:
        return V[v] - 1.08574398
    else:
        return V[v] + F13 + x*(f-2) - offset

if __name__ == "__main__":
    F = np.zeros(4)
    print(" N      F1          F2         F3")
    for v in range(1):
        for N in range(1, 100, 2):
            F[2] = rouille_(v, N, N)
            for J in range(N-1, N+2, 2):
                f = N - J + 2
                F[f] = rouille_(v, N, J)
            print("{:2d} {:10.5f} {:10.5f} {:10.5f}"
                  .format(N, F[1], F[2], F[3]))
