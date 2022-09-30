#!/usr/bin/env python2.7
""" 
 Fit Dunham polynomial    EvJ = \Sum Ykl (v+1/2)^k [J(J+1)]^l

 Stephen.Gibson@anu.edu.au
 10 July 2014
"""

from numpy import *
from matplotlib.pylab import *
from scipy.optimize import curve_fit

def Dunham (v,Y):
    x = v+1/2 
    sum = 0.0
    for y in Y[:0:-1]: sum = (sum+y)*x
    return sum+Y[0]

def DunV (v,*Y):  # there must be a better way?
    ans = []
    for vv in v: ans.append(Dunham(vv,Y))
    return ans

def pwr(x):
    return int(floor(log10(x))) if x > 0 else 1

def fmt (num,err):
    ''' format xxx+-yyy  number as xxx(yyy) '''
    ints = abs(pwr(num))
    decpl = pwr(err)
    adecpl = abs(decpl)
    if err < 1:
        f = '%'+("%d.%d" % (ints+adecpl,adecpl)) + "f(%d)"
        fmt = f % (num,round(err/(10**decpl)))
    else:
        f = '%'+("%d.%d" % (ints,0)) + "f(%d)"
        fmt = f % (num,round(err))
    return fmt

def LGcorr (v,Gv,label='X'):
    "correct Tv values for (2xlambda/3 - gamma)"
    if label=='X':
# Green and Wheeler JCP 104, 848 (1996) Table II
# v" = 0,..,5,7
        lam = array([11.7931,11.8659,11.937,12.021,12.091,12.163,12.247])
        gam = array([-0.007157,-0.007148,-0.00593,-0.00066,-0.000722,-0.007283,-0.00962])
    elif label=='B':
# Green and Wheeler JCP 104, 848 (1996) Table IV
# v" = 0,..,6
        lam = array([2.877,3.530,2.42,3.15,3.45,2.58,1.42])
        gam = array([-0.280,-0.0185,-0.0072,-0.0028,-0.0089,-0.0009,0.0029])
    else:
# Green and Wheeler JCP 104, 848 (1996) Table V
# v'= 2,...,12
        lam = array([5.343,6.34,6.001,5.342,4.98,4.52,5.77,5.54,6.600,5.30,0])
        gam = array([0.122,0.199,0.63,0.148,0.385,0.037,0.074,0.28,-0.042,0.343,0])
    twothirds = 2*lam/3.0-gam
    return Gv - twothirds

#--- main ----
fn = input ("v Gv Bv filename? ")
label = fn #fn[-5:-4]
print(label)

vv,Gv,Bv = loadtxt (fn,unpack=True)
#vv,Gv,Bv,lam,mu,oldGv = loadtxt (fn,unpack=True)
if vv[0] < 0: # drop v=-1/2 values
    vc=vv[1:]; Gc=Gv[1:]; Bc=Bv[1:]  
else:
    vc=vv; Gc=Gv; Bc=Bv

for i,vi in enumerate(vc): print("%g %g %g" % (vi,Gc[i],Bc[i]))
print() 

# apply (2xlambda/3 - gamma) correction
#Gc = LGcorr (vc,Gc,label)
#print Gc

# Gv - fit Dunham polyn to determine G(v=-1/2)
# limit to lowest 3 levels
par, err = curve_fit (DunV, vc[:4], Gc[:4], p0=(-300,0.1,-0.01)) 
YG = par; eYG = diag(err)
for i,y in enumerate(YG):
    print("Y%d0 = %s" % (i,fmt(y,eYG[i])))
G12 = Dunham(-0.5,YG)
print("G(v=-0.5) = %g" % G12)

print() 
# Bv - fit Dunham polyn to determine B(v=-1/2)
par, err = curve_fit (DunV, vc[:4], Bc[:4], p0=(1,0.1,-0.1)) 
YB = par; eYB = diag(err)
for i,y in enumerate(YB):
    print("Y%d1 = %s" % (i,fmt(y,eYB[i])))
print("B(v=-0.5) = %g" % Dunham(-0.5,YB))

Y00 = YG[0];  Y01 = YB[0]
G0 = Gc[0]
print("\nConstants referenced to G(v=0)")
print("%2.1f %8.5f %9.7f" % (-0.5,G12-G0,Y01))
for i,vi in enumerate(vc): print("%2.1f %8.5f %9.7f" % (vi,Gc[i]-G0,Bc[i]))

subplot(211)
title (fn)
plot (vv,Gv,'o')
vr = arange (-0.5,vv[-1]+1,0.1)
plot (vr,Dunham(vr,YG))
xlabel("$v$")
ylabel("$G_v$")
subplot(212)
plot (vv,Bv,'o')
plot (vr,Dunham(vr,YB))
xlabel("$v$")
ylabel("$B_v$")
show()
