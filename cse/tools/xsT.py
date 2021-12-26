import numpy as np
import os
import glob
import re
import gzip
from cse.tools.intensity import Boltzmann

def total_cross_section(T, sym='f', outfile=None, dirpath='./',
                        suffix='.dat.gz', verbose=False):
    """ Evaluate total cross section from partial cross section files
        through Boltzmann sum.

        math::Σ σ_v"J" x (2J"+1)exp(-E"hc/kT)/Q_v"J"

        Assumes file names of the form 'xs_E"_J'_J"_branch'+suffix with
        header: '# E" PEC0 PEC1 ...' 
        data:  wavenumber xs0 xs1 ...    cross sections for each channel

    Parameters
    ----------
    T : float 
        gas temperature in Kelvin.

    sym : str
        cross section files to be included in the summation. 
        'f' the f-symmetry files 'xs_*2.dat.gz' [default]
        'e' the e-symmetry files 'xs_*[1, 3].dat.gz'
        otherwise any glob.glob(sym) string.

    outfile: str
        default - no file.

    dirpath : str
        path to the directory containing the cross section files.

    suffix : str
        cross section datafile suffix. Default '.dat.gz'.

    Returns
    -------
    wav : numpy array of floats
        transition energies in wavenumber.
    
    xstotal: numpy array of floats
        total and channel cross sections for each column.

    Q: float
        partition function, for the summation.

    """

    xstotal = None
    Q = 0.0

    if len(sym) > 0:
        if sym == 'f':
            symf = 'xs_*2'
        else:
            symf = 'xs_*[1, 3]'

    if verbose:
        print(f'{sym}-levels: filename'
              f'       E"    J\'  J"  branch   Boltz.  σ_max.'
              '-------------')

    xsfiles = os.path.join(dirpath, symf+suffix)
    for xsf in sorted(glob.glob(xsfiles), 
                      key=lambda f:int(f.strip(dirpath).split('_')[1])):

        if verbose:
            print(f'{xsf.strip(dirpath).strip("/").strip(suffix):20s}', end=' ')

        with gzip.open(xsf, 'r') as f:  # header E" pec0 pec1 pec2 ...
            header = f.readline().decode('utf8').strip()

        wn, *xs = np.loadtxt(xsf, unpack=True, dtype=float)  # partial xs
        xs = np.array(xs)

        # extract energy of J" level
        en = float(re.findall("\d+\.\d+", header)[0])

        Jd, Jdd, branch = xsf.split('_')[-3:]
        branch = branch.strip(suffix)

        if xstotal is None:
            xstotal = np.zeros_like(xs, dtype=float)

        bltz = Boltzmann(en, int(Jdd), T)
        Q += bltz # partition function

        xstotal += xs*bltz   # each channel

        if verbose:
            print(f'{en:8.3f}  {Jd:2} {Jdd:2} {branch} {bltz:8.3f} '
                  f'{xs.sum(axis=0).max():8.3e}')

    if xstotal is None:
        print(f'error: no files in directory "{xsfiles}"')
        exit()

    xstotal[0] = xstotal[1:].sum(axis=0)  # total stored in closed channel
    xstotal /= Q

    if outfile is not None:
        if '/' not in outfile:
            outfile = os.path.join(dirpath, outfile)

        m = re.search(r'[a-z]', header, re.I) 
        if m is not None:
            header = header[m.start():]
            sep = header.find(' ')
            if sep > 0:
                header = header[sep:]
        else:
            header = ' '

        np.savetxt(outfile, np.column_stack((wn, *xstotal[:])),
                   fmt='%8.4f'+' %10.3e'*xstotal.shape[0],
                   header='wavenumber   total' + header.replace(' ', ' '*8))

    return wn, xstotal, Q
