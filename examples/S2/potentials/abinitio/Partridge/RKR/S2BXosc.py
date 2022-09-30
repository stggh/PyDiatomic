import numpy as np
import cse

S2expt = np.loadtxt("S2-BXosc-Glenn.dat")
S2Part = np.loadtxt("fosc-Partridge.dat", unpack=True)
S2Part = (S2Part[0], S2Part[1]*10)

S2 = cse.Xs('S32', VTi=['X3S-1rkr.dat'], VTf=['B3S-1rkr.dat'],
            dipolemoment=['../../DBXA.dati'], eni=365)

S2cse = cse.Xs('S32', VTi=['../../../X/X3S-1.dat'], 
               VTf=['../../../B/B3S-1.dat'],
               dipolemoment=['../../DBXA.dati'], eni=365)

S2brl = cse.Xs('S32', VTi=['../../../X/X3S-1.dat'], 
               VTf=['../../../Brenton/B_3S-1_temp14.dat'],
               dipolemoment=['../../DBXA.dati'], eni=365)


def G(v):
    we = 434
    wexe = 2.75
    return we*(v+1/2) - wexe*(v+1/2)**2

bands = np.array([31835+G(v)-S2.gs.cm for v in range(7)])

S2.calculate_xs(transition_energy=bands) 
S2cse.calculate_xs(transition_energy=bands) 
S2brl.calculate_xs(transition_energy=bands) 

print("fosc X 10^4")
print(" v'    Glenn(exp)  Partx3    PartRKR      CSE     CSE(B_3S-1_temp14)")
for v, fexp in S2expt:
   v = int(v) 
   print("{:2d} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}"
         .format(int(v), fexp, S2Part[1][v]*3, S2.xs[v, 0]*1.0e4, 
                 S2cse.xs[v, 0]*1.0e4, S2brl.xs[v, 0]*1.0e4))
