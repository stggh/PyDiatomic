import numpy as np
import cse
import matplotlib.pyplot as plt

#####################################################################
#
# fit Wei analytical function to a potential energy curve
#
#             [ 1 - exp(-b(r-re))  ]^2
#   V(r) = De [ ----------------   ]   + Te       |h| < 1
#             [ 1- h exp(-b(r-re)) ]
#
#####################################################################

B = cse.Cse('O2', dirpath='potentials', VT=['B3S-1.dat'])
B.levels(5)
print(B)

print('B-state Wei analytical fit:')
subr = np.logical_and(B.R > 1.3, B.R < 2.8)
r = B.R[subr]
v = B.VT[0, 0][subr]*B._evcm
Re = r[v.argmin()]
voo = B.VT[0, 0][-1]*B._evcm
De = voo - B.VT[0, 0].min()*B._evcm

res = cse.tools.analytical.Wei_fit(r, v, re=Re, De=De, voo=voo, 
                                   adjust=['re', 'De', 'b', 'h'],
                                   verbose=True)

WB = cse.tools.analytical.Wei(B.R, *list(res.paramdict.values()))

Y = cse.Cse('O2', VT=[(B.R, WB)])
Y.levels(5)
print(Y)

# plots ----------------
plt.plot(B.R, B.VT[0, 0]*B._evcm, '--', label=r'O$_2$ $B^3\Sigma_u^-$')
plt.plot(Y.R, Y.VT[0, 0]*Y._evcm, label='Wei fit')
plt.annotate(res.fitstr, (2.5, 50000), fontsize='small')

plt.axis([0.8, 5, 47500, 65000])
plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
plt.xlabel(r'internuclear distance ($\AA$)')
plt.ylabel(r'potential energy (cm$^{-1}$)')
plt.legend()

plt.savefig('figures/example_Wei_fit.svg')
plt.show()
