import numpy as np
import cse
import re
from scipy.interpolate import splrep, splev
from scipy.optimize import least_squares
from scipy.linalg import svd


class Model_fit():
    """ Least-sqaures fitting CSE model parameters to any experimental
        data.

        CSE model parameters:
          - electronic potential energy curves,
          - electric transition dipole moment
          - couplings

        Data dict to fit to:
        'xs' : {any_label, ([wavenumber], [cross section])}
        'Bv' : {statelabel, ([v], [Bv])}
        'Gv' : {statelabel, ([v], [Gv])}
        'width' : {any_label, (wn0, wn1, width, J)}

    Attributes
    ----------

    """

    def __init__(self, csemodel, data2fit, VT_adj={}, coup_adj={}, etdm_adj={},
                 verbose=True):
        """
        Parameters
        ----------
        data2fit: dict(Gv=v_Gv_arrays, Bv=v_Bv_arrays,
                        exptxs=wavenumber_cross_section_arrays)
        # potential energy curves -------
        VT_adj: {
            'ΔR':float,  # radial shift
            'ΔV':float,  # energy shift
            # parameters for Wei analytical PEC, applied Rm...Rn
            'Wei':{'re':re, 'De':De, 'voo':voo, 'b':b, 'h':h, 'Rm':Rm, 'Rn':Rn},
            # Julienne (open channel), applied Rm...Rn
            'Julienne':{'Mx':Mx, 'Rx':Rx, 'Vx':Vx, 'Voo':Voo, 'Rm':Rm, 'Rn':Rn),
            'spline':(R₀, R₁, ..., Rn),  # radial positions of knots
            )

        # coupling scale factor -------
        coup_adj: {'state0_state1':float, 'state0_state2':float}  # etc.

        # electric dipole transition moment -------
        etdm_adj: {'fsl0_isl0':float, 'fsl1_isl0':float)  # etc.
        """

        self.verbose = verbose

        self.csemodel = csemodel  # CSE Transition instance
        self.R = csemodel.us.R  # internuclear distance
        self._evcm = csemodel.us._evcm  # conversion eV to cm⁻¹

        # interaction matrix
        self.VT_orig = csemodel.us.VT.copy()
        # diagonal elements = PECs
        self.VTd_orig = np.diagonal(self.VT_orig).T.copy()*self._evcm

        # electric diple transition moments
        self.dipolemoment_orig = csemodel.dipolemoment.copy()

        # CSE model parameters to fit ---------------------
        self.VT_adj = VT_adj  # PEC adjustments
        self.coup_adj = coup_adj  # couplings
        self.etdm_adj = etdm_adj  # electronic transition dipole moment

        # experimental data --------------------------
        self.data2fit = data2fit  # experimental data sets

        # least-squares fitting -------------------------------
        self.parameter_pack()  # parameter list

        self.fit()  # least-squares fit
        self.fiterrors()  # parameter error estimates
        self.print_result()

    def parameter_pack(self):
        """ build a parameter list, required for `least_squares()`.

        """
        lsqpars = []
        bounds_min = []
        bounds_max = []

        def addparam(param):
            if isinstance(param, tuple):
                par, bmin, bmax = param
            else:
                par, bmin, bmax = param, -np.inf, np.inf

            lsqpars.append(par)
            bounds_min.append(bmin)
            bounds_max.append(bmax)

        for state, state_dict in self.VT_adj.items():
            for param, param_dict in state_dict.items():
                match param:
                    # single value
                    case 'ΔR' | 'ΔV':
                        addparam(param_dict)
                    # multi-value - To do store parameters
                    case 'Wei' | 'Julienne':
                        for k, v in param_dict.items():
                            if k not in ['Rm', 'Rn']:
                                addparam(v)
                    case 'spline':
                        for x in param_dict:  # actually an array
                            addparam(1.)

        for v in self.coup_adj.values():
            addparam(v)
        for v in self.etdm_adj.values():
            addparam(v)

        self.lsqpars = lsqpars
        self.bounds = (bounds_min, bounds_max)

    def parameter_unpack(self):
        """ extract new CSE model parameters from values of the least_squares
            parameter list.

        """
        lsqpars = list(self.lsqpars)

        R = self.R
        VTd = self.VTd_orig.copy()

        # potential energy curve adjustments --------------------
        for state, state_dict in self.VT_adj.items():
            indx = self.csemodel.us.statelabel.index(state)
            for param, value in state_dict.items():
                match param:
                    # single value
                    case 'ΔR':
                        value = lsqpars.pop(0)
                        spl = splrep(R-value, VTd[indx])
                        VTd[indx] = splev(R, spl)

                    case 'ΔV':
                        value = lsqpars.pop(0)
                        VTd[indx] += value

                    # multi-value
                    case 'Wei':
                        subR = np.logical_and(R >= value['Rm'],
                                              R <= value['Rn'])
                        VTd[indx][subR] = cse.tools.analytical.Wei(R[subR],
                                                                   **value)
                        lsqpars = lsqpars[len(value)-2:]

                    case 'Julienne':
                        subR = np.logical_and(R >= value['Rm'],
                                              R <= value['Rn'])

                        VTd[indx] = cse.tools.analytical.Julienne(R[subR],
                                                                  **value)
                        lsqpars = lsqpars[len(value)-2:]

                    case 'spline':
                        spl = splrep(value, lsqpars[:len(value)])

                        subR = np.logical_and(R >= value[0], R <= value[-1])
                        VTd[indx][subR] *= splev(R[subR], spl)

                        lsqpars = lsqpars[len(value):]

            # returned modified PECs to csemodel
            self.csemodel.us.VT[indx, indx] = VTd[indx]/self._evcm  # diagonal

        # coupling -----------------------------------------
        for lbl, scaling in self.coup_adj.items():
            scaling = lsqpars.pop(0)
            state1, state2 = re.split('<->', lbl)
            i = self.csemodel.us.statelabel.index(state1)
            j = self.csemodel.us.statelabel.index(state2)
            coupling = self.VT_orig[i][j]
            self.csemodel.us.VT[i][j] = self.csemodel.us.VT[j][i]\
                                      = coupling*scaling

        # etdm -------------------------------------------------
        dipolemoment = self.dipolemoment_orig.copy()

        for lbl, scaling in self.etdm_adj.items():
            scaling = lsqpars.pop(0)
            f, i = re.split('<-', lbl)
            indxf = self.csemodel.us.statelabel.index(f)
            indxi = self.csemodel.gs.statelabel.index(i)
            dipolemoment[:, indxi, indxf] *= scaling

        # return modified dipolemoment array to csemodel
        self.csemodel.dipolemoment = dipolemoment

    def print_result(self):
        lsqpars = list(self.result.x)
        stderr = list(self.result.stderr)

        print('\n\nModel fitted parameters')
        print('Potential energy curves ----------------------------')
        for state, state_dict in self.VT_adj.items():
            print(f'{state:10s} ', end='')
            for param, param_dict in state_dict.items():
                match param:
                    # single value
                    case 'ΔR' | 'ΔV':
                        print(f'{param:10s} ', end='')
                        print(f'{lsqpars.pop(0):5.3g}±{stderr.pop(0):.3g} cm⁻¹')
                        print(f'{" ":10s}', end='')
                    # multi-value - To do store parameters
                    case 'Wei' | 'Julienne':
                        print(f'{param:5s} ', end='')
                        for k, v in param_dict.items():
                            if k not in ['Rm', 'Rn']:
                                print(f'{k:5s}: {lsqpars.pop(0):5.3g}±'
                                      f'{stderr.pop(0):.3g}')
                                print(f'{" ":10s}', end='')
                    case 'spline':
                        print(f'{param:6s} r(Å)  scaling')
                        for r in param_dict:  # actually an array
                            print(f'{" ":16s}', end='')
                            print(f'{r:5.3g} ', end='')
                            print(f'{lsqpars.pop(0):5.3f}±{stderr.pop(0):.3f}')
            print()

        if self.coup_adj:
            print('\nCoupling - ------------------------- (x) scaling factor'
                  '--')
            for lbl, v in self.coup_adj.items():
                print(f'{lbl:>10s} {" ":6s} '
                      f'{lsqpars.pop(0):5.3g}±{stderr.pop(0):.3g}')

        if self.etdm_adj:
            print('\nElectronic transition dipole moment '
                  '- (x) scaling factor --')
            for lbl, v in self.etdm_adj.items():
                print(f'{lbl:>10s} {" ":8s} '
                      f'{lsqpars.pop(0):5.3g}±{stderr.pop(0):.3g}')

        print()

    def fiterrors(self):
        ''' from:
            https://stackoverflow.com/questions/42388139/how-to-compute-standard-deviation-errors-with-scipy-optimize-least-squares

        '''

        U, s, Vh = svd(self.result.jac, full_matrices=False)
        tol = np.finfo(float).eps*s[0]*max(self.result.jac.shape)
        w = s > tol
        cov = (Vh[w].T/s[w]**2) @ Vh[w]  # robust covariance matrix
        chi2dof = np.sum(self.result.fun**2) / \
                        (self.result.fun.size - self.result.x.size)
        cov *= chi2dof
        self.result.stderr = np.sqrt(np.diag(cov))

    def residual(self, pars):
        self.lsqpars = pars.copy()
        self.parameter_unpack()  # resets csemodel with modified parameters

        self.diff = []
        for channel, data_dict in self.data2fit.items():
            chnl_indx = self.csemodel.us.statelabel.index(channel)
            for data_type, data in data_dict.items():
                match data_type:
                    case 'xs':
                        wavenumber, xs = data
                        self.csemodel.calculate_xs(transition_energy=wavenumber)
                        self.csexs = self.csemodel.xs[:, chnl_indx]
                        self.diff.append((self.csexs-xs)*1e19)
                    case 'peak':
                        pos = data
                        wavenumber = np.arange(pos-10, pos+10, 0.1)
                        self.csemodel.calculate_xs(transition_energy=wavenumber)
                        self.peakxs = self.csemodel.xs[:, chnl_indx]
                        self.peak = wavenumber[self.peakxs.argmax()]
                        self.diff.append(np.array((self.peak-pos)*100))

        self.diff = np.ndarray.flatten(np.asarray(self.diff))
        self.sum = self.diff.sum()

        if self.verbose:
            print(f'{pars} {self.sum:g}')
        else:
            print('.', end='')

        return self.diff

    def fit(self):
        self.result = least_squares(self.residual, self.lsqpars,
                                    bounds=self.bounds, method='trf',
                                    x_scale='jac', diff_step=0.1)
