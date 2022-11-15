import numpy as np
import cse
import glob
import os
import gzip
import re
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import simps
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

######################################################################
#
# Analyse cross section files 'dirpath'/xs_E"_J'_J"_{branch}/dat.gz
#  to provide transition energies, linewidths.
#   cf with Harvard/Vijre data where available.
#
#  Stephen.Gibson@anu.edu.au - Nov 2022
#
######################################################################

def parse(dirpath):  # extract molecule and v' from directory name
    par = dirpath.split('_')
    if len(par) > 2:
        vd = np.arange(int(par[1]), int(par[2])+1, 1)
    else:
        vd = [int(par[-1]),]
    miso = re.search(r'\d', par[0])
    iso = par[0][miso.start():]

    return iso, vd[0]

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

def analyse(dirpath, suffix='.dat.gz', lambda23=3, wn_offset=3):
    Ncol = np.zeros(20, dtype=int)
    wn_table = np.zeros((20, 6))
    fwhm_table = np.zeros((20, 6))
    xs_table = np.zeros((20, 6))

    xsfiles = os.path.join(dirpath, 'xs*'+suffix)
    for xsfile in sorted(glob.glob(xsfiles),
                      key=lambda f:int(f.strip(dirpath).split('_')[1])):

        with gzip.open(xsfile, 'r') as f:  # header E" pec0 pec1 pec2 ...
            header = f.readline().decode('utf8').strip()

        wn, *xs = np.loadtxt(xsfile, unpack=True, dtype=float)  # partial xs
        wn -= wn_offset   # gobal wavenumber shift of the calculated spectrum

        xs = np.array(xs).T
        xst = xs.sum(axis=1)
        # shift f-levels by an additional 2λᵥ/3
        if '2.dat' in xsfile and abs(lambda23) > 0:
            spl = splrep(wn+lambda23, xst)
            xst = splev(wn, spl)
        dw = wn[1] - wn[0]

        # extract energy of J" level
        en = float(re.findall("\d+\.\d+", header)[0])

        Jd, Jdd, branch = xsfile.split('_')[-3:]
        branch = branch.strip(suffix)
        ΔN, ΔJ, Fd, Fdd = decode(branch)
        Ndd = int(Jdd) + Fdd - 2

        peaks, _ = find_peaks(xst, height=xst.max()/10)
        if len(peaks) == 0:
            continue

        FWHM, xsx, _, _ = peak_widths(xst, peaks, rel_height=1/2)
        row = Ndd // 2
        col = branches.index(branch)
        Ncol[row] = Ndd
        pk0 = peaks[0]
        wnJ = wn[pk0]  # select lowest energy
        fwhmJ = FWHM[0]*dw
        wn_table[row, col] = wnJ
        fwhm_table[row, col] = fwhmJ

        xs_table[row, col] = simps(xst[pk0-5:pk0+5], wn[pk0-5:pk0+5])

        row += 1
    return Ncol[:row], wn_table[:row], fwhm_table[:row], xs_table[:row]

def print_out(title, Ncol, table,
              branches=['R1', 'R2', 'R3', 'P1', 'P2', 'P3']):

    title_str = (title+' (cm⁻¹) '+'-'*70)[:80]
    print(title_str)
    print('N"\\branch', end='')

    pretty = lambda br: br[0] + ['₀', '₁', '₂', '₃'][int(br[1])]
    for br in branches:
        print(f'{pretty(br):^11s}', end='')
    print()

    for N, x in zip(Ncol, table):
        print(f'  {N:2d}  ', end='')
        for col, d in enumerate(x):
            if d == 0:
                print(f' {"-":^10s}', end='')
            else:
                print(f'{d:10.3f} ', end='')
        print()
    print()

def differences(wn_table, yosh):

    table = wn_table.copy()
    try:
        yos = np.loadtxt(yosh)
        # rowmax = int(yos[:, 0].max()) // 2 + 1
        rowmax = yos.shape[0]
        table[:rowmax, :] -= yos[:, 1:]
        table = table[:rowmax-1]
        diff = table > 10 
        table[diff] = 0 
    except FileNotFoundError:
        print(f'Error: no file "{yos}"')
        exit()

    return table

def print_stats(table):
    # mean differences
    ave = []
    std = []
    for col in table.T:
        good = abs(col) < 5 #  np.logical_and(col > -5, col < 5)
        ave.append(col[good].mean())
        std.append(col[good].mean())

    print('mean:', end='')
    for a in ave:
        print(f' {a:10.3f}', end='')
    print()

    print('std: ', end='')
    for s in std:
        print(f' {s:10.3f}', end='')
    print()

    print()


def cross_section(dirpath, iso, vd, lambda23=3, wn_offset=3):
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, sharey=True)

    if vd in range(1, 13):
        isoy = {'16O2':'6l', '16O18O':'68', '18O2':'18'}
        xsexp = {79:f'data/Harvard/o{isoy[iso]}b{vd}.xsc',
                 295:f'data/Harvard/o2wb{vd}x0.xsc'}
    elif vd == 14:
        xsexp = {79:None, 295:f'Yoshino_140.csv'}
    else:
        xsexp = {79:None, 295:None}

    wn = {}
    xs = {}
    for ax, T in zip([ax0, ax1], [79, 295]): 
        for sym in ['f', 'e']:
            wn[sym], xs[sym], Q = cse.tools.xsT.total_cross_section(T, sym,
                                          outfile=f'{T}K{sym}', dirpath=dirpath,
                                          verbose=False)
        print(f'T = {T:.0f} Q = {Q:5.3f}')

        if xsexp[T] is not None:
            try:
                wny, xsy = np.loadtxt(xsexp[T], unpack=True)
                xsy -= xsy.min()
                if xsy.max() > 1:
                    xsy *= 3e-18
                ax.plot(wny, xsy, label=f'{xsexp[T][:-4]} v\'={vd}')
            except:
                xsy = None
        else:
            xsy = None

        # e-levels global offset
        wn['e'] -= wn_offset
        xse = xs['e'][0]

        # shift f-levels by an additional 2λᵥ/3
        wn['f'] -= wn_offset
        wavenumber = wn['e']
        spl = splrep(wn['f']+lambda23, xs['f'][0])
        xsf = splev(wavenumber, spl)

        xst = xse + xsf
        np.savetxt(f'{dirpath}/{T}K', np.column_stack((wavenumber, xst)),
                   fmt='%8.3f %10.5e')

        if xsy is not None:
            subr = np.logical_and(wn['f'] >= wny[0], wn['f'] <= wny[-1])
            sf = xsy.max()/xst[subr].max()
            print(f'scaling factor = {sf:g}\n')
            xst *= sf

        ax.plot(wavenumber, xst, label='PyDiatomic')
        #  ax.plot(wn['f'], xs['f'][0], '--', label='f')
        #  ax.plot(wn['e'], xs['e'][0], '--', label='e')

        ax.set_title(f'v\'={vd} {T}K')
        ax.set_ylabel(r'cross section (cm$^{2}$)')
        ax.legend(fontsize='small', labelspacing=0.1)

    ax1.set_xlabel(r'transition energy (wavenumber cm$^{-1}$)')

    plt.savefig('figures/O2_SRB_analyse_xs.svg')
    plt.show()

# main --------------------------------------
dirpath = 'Ax16O2_12'
wn_offset = 3
lambda23 = 3 

iso, vd = parse(dirpath)
branches = ['R1', 'R2', 'R3', 'P1', 'P2', 'P3']

Ncol, wn_table, fwhm_table, xs_table  = analyse(dirpath, lambda23=lambda23,
                                                wn_offset=wn_offset)

print_out(f'({vd}, 0) transition energies', Ncol, wn_table)

expt = {12:'data/Harvard/Yosh_120.dat', 0:'data/Vijre/Ubachs_00.dat',
         2:'data/Vijre/Ubachs_20.dat'}
if vd in expt.keys(): 
    diff_table = differences(wn_table, expt[vd])
    print_out(f'({vd},0) transition energies difference to {expt[vd][:-4]}', 
              Ncol, diff_table)
    print_stats(diff_table)

print_out(f'({vd},0) linewidths', Ncol, fwhm_table)

print_out(f'({vd},0) integrated cross section', Ncol, xs_table*1e19)

cross_section(dirpath, iso, vd=vd, lambda23=lambda23, wn_offset=wn_offset)
