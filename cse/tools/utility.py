import numpy as np
from scipy.signal import find_peaks, peak_widths

def peaks(wavenumber, xs, dw=None, width=(0.1, 400), verbose=True):
    if dw is None:
        dw = wavenumber[1] - wavenumber[0]

    width = (width[0]/dw, width[1]/dw)
    pks, _ = find_peaks(xs, width=width)
    widths = peak_widths(xs, pks, rel_height=1/2)[0]*dw
    pos = wavenumber[pks]

    if verbose:
        print('wavenumber (cm⁻¹)    FWHM (cm⁻¹)     height (cm²)')
        for p, w, i in zip(pos, widths, pks):
            s = ''
            if p < 100000:
                s = ' '
            print(f' {s}{p:,.2f}         {w:8.3f}       {xs[i]:8.1e}')
        print()

    return pos, widths, pks


def turning_points(state, Xzpe=0, vmax=5):
    evcm = state._evcm
    Voo = state.VT[0, 0, -1]*evcm  # dissociation limit

    Re = state.R[state.VT[0, 0].argmin()]
    inner = state.R < Re
    outer = state.R > Re

    if len(state.results) == 0 or\
       max(state.results, key=state.results.get)+1 < vmax:
        state.levels(vmax)

    tp = []
    for v, (Tv, *_) in state.results.items():
        if Tv < Voo:  # bound
            ri = state.R[inner][np.abs(state.VT[0, 0][inner]-Tv/evcm).argmin()]
            ro = state.R[outer][np.abs(state.VT[0, 0][outer]-Tv/evcm).argmin()]
            Tv -= Xzpe
            tp.append((v, Tv, ri, ro)) 

    return tp

def osc2FCF(osc, bands):
    ''' oscillator strength to Franck-Condon factor conversion.

    '''

    return osc*1e6/bands/3
