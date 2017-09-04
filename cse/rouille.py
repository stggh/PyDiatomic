#####################################################
#   O2 X-state energy levels
#   G. Rouille, G. Millot, R. Saint-Loup, and H. Berger
#   J. Mol. Spectrosc. 154, 372-382 (1992). 
#
#   Stephen.Gibson@anu.edu.au August 2010
#####################################################

from numpy import *

def A_(J, B, D, H, lamb, lambdap, lambdapp, mu, mup, mupp, muppp): 
   return (2*J + 1) * (B - 2*D*(J*J + J+1) + H*(3*J**4+6*J**3+12*J*J+10*J+4)\
         -0.5*mu-0.5*mup*(J*J+J+4)-0.5*mupp*(J**4+2*J**3+13*J*J+12*J+8)\
         +1.5*muppp)-(lamb+lambdap*(7*J*J+7*J+4)/3.0+lambdapp*(11*J**4+\
         22*J**3+39*J*J+28*J+8)/3.0)/(2*J+1.0)

def D_ (J,lamb,lambdap,lambdapp):
   return (lamb+lambdap*(J*J+J+1)+lambdapp*(J**4+2*J**3+7*J*J+6*J+2))*2.0/(2*J+1.0)

def F13split (J,B,D,H,lamb,lambdap,lambdapp,mu,mup,mupp,muppp):
    a = A_(J,B,D,H,lamb,lambdap,lambdapp,mu,mup,mupp,muppp)
    d = D_(J,lamb,lambdap,lambdapp)
    return sqrt(a*a + J*(J+1)*d*d)

def F2level (J,B,D,H,lamb,lambdap,lambdapp,mu,mup,mupp,muppp):
    x = J*(J+1)
    return B*x - D*x*x + H*x*x*x + 2.0*lamb/3.0 + 2.0*lambdap*x/3.0 +\
           2.0*lambdapp*x*x/3.0 - mu - mup*x -mupp*x*x + muppp

def F13av (J,B,D,H,lamb,lambdap,lambdapp,mu,mup,mupp,muppp):
    return B*(J*J+J+1)-D*(J**4+2*J**3+7*J*J+6*J+2)+\
           H*(J**6+3*J**5+18*J**4+31*J**3+33*J*J+18*J+4)-\
           lamb/3.0-lambdap*(J*J+J+4)/3.0-\
           lambdapp*(J**4+2*J**3+13*J*J+12*J+8)/3.0-1.5*mu-\
           0.5*mup*(7*J*J+7*J+4)-0.5*mupp*(11*J**4+22*J**3+39*J*J+28*J+8)+\
           0.5*muppp*(2*J*J+2*J+5)

def rouille (v,N,J):
    V = [0.0, 1556.36103, 3089.17317, 4598.76637, 6085.10234,
         7548.36948, 8988.7386]
    B = [1.43767953, 1.42186007, 1.40612292, 1.39042964,
         1.37477988, 1.35921426, 1.34366771]
    D = [4.83984515e-6, 4.83955885e-6, 4.83666796e-6, 4.83955311e-6,
         4.84017091e-6, 4.84084911e-6, 4.84513823e-6]
    H = [2.8e-12, 2.8e-12, 2.8e-12, 2.8e-12, 2.8e-12, 2.8e-12, 2.8e-12]
    lamb = [1.984751322, 1.98957894, 1.99440656, 1.99923418, 1.99923418,
            1.99923418, 1.99923418]
    lambdap = [1.94521e-6, 2.10924e-6, 2.2733e-6, 2.4374e-6, 2.4374e-6,
               2.4374e-6, 2.4374e-6]
    lambdapp = [1.103e-11, 1.103e-11, 1.103e-11, 1.103e-11, 1.103e-11,
                1.103e-11, 1.103e-11]
    mu = [-8.425390e-3, -8.445771e-3, -8.466152e-3, -8.486533e-3, -8.486533e-3,\
          -8.486533e-3, -8.486533e-3]
    mup = [-8.106e-9, -8.264e-9, -8.42e-9, -8.58e-9, -8.58e-9, -8.58e-9,
           -8.58e-9]
    mupp = [-4.7e-14, -4.7e-14, -4.7e-14, -4.7e-14, -4.7e-14, -4.7e-14,
            -4.7e-14]
    muppp = [0, 0, 0, 0, 0, 0, 0]

    f = N - J + 2

    if v > len(V): return -1.0
    if J < 0: return -1.0
    if N==J : 
        return V[v]+F2level(N,B[v],D[v],H[v],lamb[v],lambdap[v],lambdapp[v],\
               mu[v],mup[v],mupp[v],muppp[v]) - 1.3316

    x  = F13split(J,B[v],D[v],H[v],lamb[v],lambdap[v],lambdapp[v],mu[v],\
                  mup[v],mupp[v],muppp[v])
    F13 = F13av(J,B[v],D[v],H[v],lamb[v],lambdap[v],lambdapp[v],mu[v],\
                                            mup[v],mupp[v],muppp[v])
    return V[v]-1.08574398 if J==0 and N==1 else V[v]+F13+x*(f-2)-1.3316

#----- main -------
if __name__ ==  "__main__" : 
    F = zeros(4)
    print(" N     F1         F2         F3")
    for v in range(0, 1):
        for N in range(1, 15, 2):
            F[2] = rouille(v, N, N)
            for J in range(N-1, N+2, 2):
                f = N - J + 2
                F[f] = rouille(v, N, J)
            print("{:2d} {:10.5f} {:10.5f} {:10.5f}".
                  format(N, F[1], F[2], F[3]))
