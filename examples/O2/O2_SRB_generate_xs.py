import numpy as np
import cse
import matplotlib.pyplot as plt
import time
import scipy.constants as C
import pathlib
from dataclasses import dataclass
from scipy.signal import find_peaks, peak_widths

######################################################################
#
# O2 B-X (v'-0) Schumann-Runge band - full fine-structure calculation
#
#      Evaluates coupled-channel cross sections for 
#      (B³Σ <-> ⁵Π, ³Σ⁺, ³Π, ¹Π) <- X³Σ  transitions
#
# Stephen.Gibson@anu.edu.au - Dec 2021
#
######################################################################

# CSE models ==================================================
def cse_model(iso, lun=2.005, sf=1.):
    # Xf - uncoupled ground state - f-symmetry rotational levels -----------
    O2Xf = cse.Cse(iso, VT=['potentials/X3S-1.dat'])

    # Xe - coupled ground state - e-symmetry levels -----------
    O2Xe = cse.Cse(iso, dirpath='potentials', suffix='.dat',
                   VT=['X3S-1', 'X3S0', 'b1S0'], coup=[-lun, 0, 229])

    # B - uncoupled upper state
    O2B = cse.Cse(iso, VT=['potentials/B3S-1.dat'])

    # Bf - coupled upper states - f-levels -----------
    O2Bf = cse.Cse(iso, dirpath='potentials', suffix='.dat',
                   VT=['B3S-1', '5P1', '3S+1', '3P1', '1P1'],
                   coup=[74*sf, 38*sf, 29*sf, 28*sf, *([0]*6)])

    # Be - coupled upper states - e-levels -----------
    O2Be1 = cse.Cse(iso, dirpath='potentials', suffix='.dat',
                    VT=['B3S-1', '5P1', '3S+1', '3P1', '1P1',
                        'B3S-0', '5P0', '3P0'],
                    coup=[74*sf, 38*sf, 29*sf, 28*sf, -lun, *([0]*20), 74*sf, 38*sf, 0])

    # Fix me! - this instance required for correct wavefunction phase
    O2Be2 = cse.Cse(iso, dirpath='potentials', suffix='.dat',
                    VT=['B3S-1', '5P1', '3S+1', '3P1', '1P1',
                        'B3S-0', '5P0', '3P0'],
                    coup=[74*sf, 38*sf, 29*sf, 28*sf, lun, *([0]*20), 74*sf, 38*sf, 0])

    # B-X transition instances -------
    O2 = {}
    O2['f'] = cse.Transition(O2Bf, O2Xf, dirpath='transitionmoments',
                             dipolemoment=['dipole_b_valence.dat', *([0]*4)])

    O2['e1'] = cse.Transition(O2Be1, O2Xe, dirpath='transitionmoments',
                              dipolemoment=['dipole_b_valence.dat', *([0]*12),
                                            'dipole_b_valence.dat', *([0]*10)])

    O2['e2'] = cse.Transition(O2Be2, O2Xe, dirpath='transitionmoments',
                              dipolemoment=['dipole_b_valence.dat', *([0]*12),
                                            'dipole_b_valence.dat', *([0]*10)])
    return O2, O2B, O2Xf

def decode(br):   # evaluate quantum number changes from branch label
    ΔN = ord(br[0]) - ord('Q')
    if br[1].isdigit():
        ΔJ = ΔN
        Fdd = Fd = int(br[-1])
    else:
        ΔJ = ord(br[1]) - ord('Q')
        Fd = int(br[2])
        Fdd = int(br[-1])
    return ΔN, ΔJ, Fd, Fdd

def bandmodel(model):
    iso = model.iso
    Tmax = model.Tmax
    vds = model.vds
    vdds = model.vdds
    Nddmax = model.Nddmax
    dw = model.dw
    branches = model.branches
    lun = model.lun

    vdss = [str(x) for x in vds]
    # storage directory for v"J"J' xs files
    xsdir = f'{model.dirlabel}{iso}_'+'_'.join(vdss)
    pathlib.Path(xsdir).mkdir(exist_ok=True)

    t0 = time.time()

    # cse model - coupled transition instance, uncoupled excited and 
    #             initial states
    O2, O2B, O2Xf = cse_model(iso, lun)

    # vibrational levels - unperturbed X- and B- f-symmetry rotational level
    #                      states to provide reference energies
    O2Xf.levels(max(vdds)+4)
    G, B, D, J = list(zip(*O2Xf.results.values()))

    O2B.levels(max(vds)+4)
    Gd, Bd, Dd, Jd  = list(zip(*O2B.results.values()))

    # N" levels, depending on isopotomer
    Ndds = np.arange(1, Nddmax, 2)  # ¹⁶O₂, ¹⁸O₂:  ΔN" = 2
    if iso[-1] != '2':
        Ndds -= 1   # ¹⁶O¹⁸O: ΔN" = 1

    # photodissociation transition energy range (v'min, 0) .. (v'max, 0)
    if min(vds) > 1:
        wn = np.arange(Gd[min(vds)-1]-G[0]+20, Gd[max(vds)]-G[0]+20, dw)
    else:
        wn = np.arange(Gd[0]-G[0]-300, Gd[max(vds)]-G[0]+20, dw)

    # potential energy curve (channel) labels
    pecs_head = {}
    pecs_head['f'] = ' '.join(O2['f'].us.pecfs)
    pecs_head['e'] = ' '.join(O2['e1'].us.pecfs)

    # Boltzmann weight - only used to terminal calculation
    Boltz = 1

    # print page header
    print(' N" ', end='')
    for br in branches:
        print(f'{br:^12s}', end='')
    print()

    # loop v", N", J" to evaluate coupled B³Σ⁻ <- X³Σ⁻ transition
    for vdd in vdds:  # v"
        for Ndd in Ndds:  # N"
            print(f'{Ndd:2} ', end='')
            x = Ndd*(Ndd+1)
            Ef = G[vdd] + B[vdd]*x - D[vdd]*x*x
            vd = vds[-1]
            wn_est = Gd[vd] + Bd[vd]*x - Dd[vd]*x*x - Ef

            for branch in branches:
                ΔN, ΔJ, Fd, Fdd = decode(branch)

                Jdd = Ndd - Fdd + 2 
                Jd = Jdd + ΔJ
                Nd = Ndd + ΔN

                if Jdd < 0 or Jd < 0 or Nd < 0:
                    continue

                sym = ['e1', 'f', 'e2'][Fdd - 1]

                # estimate for ground state energy, 
                #    corrected by λ ~ 2 for e-levels
                edd = Ef - 2*abs(Fdd - 2)

                # cse calculation
                O2[sym].calculate_xs(transition_energy=wn, rotf=Jd, roti=Jdd,
                                     eni=edd, honl=True)

                # calculated eigenvalue, referenced to N"=J"=0 (virtual level)
                EvJ = O2[sym].gs.cm - G[0]

                # save cross section for each channel - a column in the file
                # B-state, column 0, is closed, has zero value
                fn = f'{xsdir}/xs_{int(EvJ)}_{Jd}_{Jdd}_{branch}.dat.gz'
                np.savetxt(fn, np.column_stack((wn, *O2[sym].xs.T)), 
                           fmt='%8.5f'+' %15.8e'*O2[sym].xs.shape[-1],
                           header=f' {EvJ:8.5f} '+ pecs_head[sym[0]])
             
                # calculation progress - print transition position
                xsJ = O2[sym].xs.sum(axis=1)  # total xs all channels
                # peaks
                peaks, _ = find_peaks(xsJ, height=xsJ.max()/10)
                if len(peaks) == 0:
                    continue   #  transition energy range has no peak

                FWHM, xsx, _, _ = peak_widths(xsJ, peaks, rel_height=1/2)
                FWHM *= dw

                # closest rovibrational peak to estimated transition energy
                pk = peaks[np.abs(wn_est - wn[peaks]).argmin()]
                wnJ = wn[pk]

                if wnJ > wn[0]:
                   print(f'{wnJ:12.3f}', end='')
                else:
                   print(f'{"-":^12s}', end='')

                Boltz = (2*Jdd+1)*np.exp(-EvJ*C.h*C.c*100/C.k/Tmax)
                # exit calculation if weak
                if Ndd > 5 and Boltz < 0.05 and branch == branches[-1]:
                    break

            print()

    print(f'Total execution time: {(time.time()-t0)/60:.1f} minutes')
    # typical timing Dell i7-9700 CPU @ 3.00GHz ~ 44 minutes, for 295K
    # NB the calculation only needs to be run once, for the Tmax required!
    print('\nCross section files "xs_E"_J\'_J"_{branch}.dat.gz'
          f'\n  written to directory: "{xsdir}"') 
    print('run `O2SRB_analyse.py` to process')

# main ===================================================
@dataclass
class Model:
    dirlabel: str   # {dirlabel}{iso}_{vd}
    iso: str        # 16O2, 16O18O, or 18O2
    Tmax: float     # highest O2 gas temperature in Kelvin
    vds: list       # [v', ... v'max]
    vdds: list      # [v", ... v"max]
    Nddmax: int     # N"max
    dw: float       # transition_energy grid spacing cm⁻¹
    branches: list  # rotational branches 
    lun: float      # L-uncoupling constant, usually 2.005


model = Model(dirlabel='Ax', iso='16O2', Tmax=295, vds=[5], vdds=[0],
              Nddmax=31, dw=0.1, lun=2.005,
              branches=[ 'R1', 'R2', 'R3', 'P1', 'P2', 'P3']) 

print('Warning this example takes some time to execute:')
print('     Typically 45 minutes on a Dell i7-9700 CPU @ 3.00GHz,')
print('     depending on the number of rovibrational transitions required for'
      ' Tmax')
print()
print('Note, the calculation need be only be run once, for the Tmax required!',
      end='\n\n') 

bandmodel(model)
