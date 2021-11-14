import numpy as np
import glob
import re
import gzip
from cse.tools.intensity import Boltzmann

def total_cross_section(T, select='f', outfile=None, dirpath='./'):
    """ Evaluate total cross section from partial cross section files
        through Boltzmann sum.

        math::Σ σ_v"J" x (2J"+1)exp(-E"hc/kT)/Q_v"J"

        Assumes file names of the form 'xs_E"_J'_J"_branch.dat.gz' with
        header: '# E" PEC0 PEC1 ...' 
        data:  wavenumber xs0 xs1 ...    cross sections for each channel

    Parameters
    ----------
    T : float 
        gas temperature in Kelvin.

    select : str
        cross section files to be included in the summation. 
        'f' is the f-symmetry files 'xs_*2.dat.gz' [default]
        'e' is the e-symmetry files 'xs_*[1, 3].dat.gz'
        otherwise any glob(select) string.

    outfile: str
        write total cross section to a file.
        'auto' = f'{T}K{select}' e.g. '295Kf'

    dirpath : str
        path to the directory containing the cross section files.

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

    if len(select) > 1:
        if select == 'f':
            select = 'xs_*2.dat.gz'
        else:
            select = 'xs_*[1, 3].dat.gz'

    for xsf in sorted(glob.glob(select), key=lambda f:int(f.split('_')[1])):

        with gzip.open(xsf, 'r') as f:  # header E" pec0 pec1 pec2 ...
            header = f.readline().decode('utf8').strip()

        wav, *xs = np.loadtxt(xsf, unpack=True, dtype=float)  # partial xs
        xs = np.array(xs)

        en = re.findall("\d+\.\d+", header)[0]  # extract energy of J" level

        Jd, Jdd, branch = xsf.split('_')[2:5]

        if xstotal is None:
            xstotal = np.zeros_like(xs, dtype=float)

        bltz = Boltzmann(en, int(Jdd), T)
        Q += bltz # partition function

        xstotal[1:, :] += xs[1:, :]*bltz   # each open channel
        xstotal[0, :] += xs[1:, :].sum(axis=0)*bltz  # total in closed channel

    xstotal /= Q

    if outfile is not None:
        if outfile == 'auto':
            outfile = f'{T}K{select}'

        np.savetxt(outfile, np.column_stack((wav, *xstotal[:])),
                   fmt='%8.4f'+' %10.5e'*xstotal.shape[0],
                   header='wavenumber'+newhead.replace(' ', ' '*5))

    return wav, xstotal, Q
