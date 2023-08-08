import numpy as np
import cse
import re
from cse.tools.analytical import Julienne, Wei, fiterrors
from scipy.interpolate import splrep, splev
from scipy.optimize import least_squares
from scipy.linalg import svd


class Model_fit():
    """ Least-squares fitting CSE model parameters to any experimental
        data.

        CSE model parameter dicts:
          VT_adj : electronic potential energy curves,
          coup_adj : couplings
          etdm_adj : electric transition dipole moment
           
        data2fit: data dict to fit:
        'xs' : {channel statelabel: ([wavenumber], [cross section])}
        'Bv' : {statelabel: ([v], [Bv])}
        'position' : {statelabel: ([v], [wavenumber of peak])}
        'width' : {statelabel: (wn0, wn1, width, J)}

    Attributes
    ----------
    result : dict
        least_squares fit attributes, with stderr for each parameter
    csemodel : cse.Transition class
        as input, with optimised parameters
    """

    def __init__(self, csemodel, data2fit, VT_adj={}, coup_adj={},
                 etdm_adj={}, method='lm', bounds_factor=0.1, verbose=True):
        """
        Parameters
        ----------
        data2fit: {'channel':{'position': wavenumber(s)_of_peak,
                              'Bv: [v, Bv] array,
                              'xs': [wavenumber, cross_section] array},
                  {'channel2':   }}

        # potential energy curves -------
        VT_adj: {PEC:
            {'ΔR':float},  # radial shift
            {'ΔV':float},  # energy shift

            # parameters for Wei analytical PEC, applied between Rm...Rn
            {'Wei':{'Re':Re, 'De':De, 'Voo':Voo, 'b':b, 'h':h, 'Rm':Rm,
                    'Rn':Rn}},

            # Julienne (open channel), applied Rm...Rn
            {'Julienne':{'Mx':Mx, 'Rx':Rx, 'Vx':Vx, 'Voo':Voo, 'Rm':Rm,
                         'Rn':Rn}},

            # R-stretching about Rₑ
            {'Rstr': {'inner':inner_scaling, 'outer':outer_scaling}},
            {'Vstr': float},  # about V∞

            # scaling
            {'spline':[R₀, R₁, ..., R₋₁]},  # radial positions of knots
            PEC is scaled by spline between R₀..R₋₁
            )

        # coupling scale factor -------
        coup_adj: {'state0<->state1':scaling, 'state0<->state2':scaling}  # etc.

        # electric dipole transition moment -------
        etdm_adj: {'fsl0<-isl0':scaling_factor, 'fsl1<-isl0':scaling_factor} 

        NB: Each parameter is scaled relative to 1.0
        """

        self.verbose = verbose
        # least_squares parameters
        self.method = method
        self.bf = bounds_factor

        # CSE model -------------------------------------------
        self.csemodel = csemodel  # CSE Transition instance

        # handy variables
        self.R = csemodel.us.R  # internuclear distance
        self._evcm = csemodel.us._evcm  # conversion eV to cm⁻¹

        # CSE model parameters to adjust ---------------------
        self.VT_adj = VT_adj  # PEC adjustments
        self.coup_adj = coup_adj  # couplings
        self.etdm_adj = etdm_adj  # electronic transition dipole moment

        # experimental data to fit ---------------------------
        self.data2fit = data2fit  # data sets
        # weight array provided? otherwise add defaults=1
        for statedict in data2fit.values():
            for data in statedict.values():
                if len(data) < 3:
                    data += np.ones_like(data[0])  # + weights to tuple

        # least-squares fitting ------------------------------
        self.fit()  # least-squares fit
        self.result.stderr = fiterrors(self.result)  # parameter error estimates

        # update csemodel
        self.residual(self.result.x)
        if verbose:
            print(self)

    def parameter_pack(self):
        """ parameter list for least-squares fitting.

        """
        self.par_count = 0

        # interaction matrix - keep original values
        self.VT_orig = self.csemodel.us.VT.copy()
        # diagonal elements = PECs in units cm⁻¹
        self.VTd_orig = np.diagonal(self.VT_orig).T.copy()*self._evcm

        # PEC parameters ------------------------------------------------
        for statelabel, par_dict in self.VT_adj.items():
            for v in par_dict.values():
                if isinstance(v, (int, float)):
                    self.par_count += 1
                else:
                    self.par_count += len(v)

            if 'Wei' in par_dict or 'Julienne' in par_dict:
                self.par_count -= 2  # 'Rm' and 'Rn' limits not param

        # Coupling ---------------------------------------------
        self.coupling = {}
        for lbl, scaling in self.coup_adj.items():
            state1, state2 = re.split('<->', lbl)
            i = self.csemodel.us.statelabel.index(state1)
            j = self.csemodel.us.statelabel.index(state2)
            # keep original value as tuple
            self.coupling[lbl] = (self.VT_orig[i][j].copy(), i, j)
            self.par_count += 1

        self.etdm = {}
        for lbl, scaling in self.etdm_adj.items():
            lblf, lbli = re.split('<-', lbl)
            indxf = self.csemodel.us.statelabel.index(lblf)
            indxi = self.csemodel.gs.statelabel.index(lbli)
            # keep original transition moment as tuple
            self.etdm[lbl] = \
              (self.csemodel.dipolemoment[:, indxi, indxf].copy(), indxi, indxf)
            self.par_count += 1

        pars = np.ones(self.par_count)
        self.lsqpars = pars
        self.bounds = (pars*(1-self.bf), pars*(1+self.bf))

    def parameter_unpack(self, pars):
        """ extract new CSE model parameters from values of the least_squares
            parameter list.

        """
        lsqpars = list(pars)

        R = self.R
        VTd = self.VTd_orig.copy()  # original PECs, diagonal interaction array

        # potential energy curve adjustments --------------------
        for state, state_dict in self.VT_adj.items():
            indx = self.csemodel.us.statelabel.index(state)

            for param, value in state_dict.items():
                match param:
                    # single value
                    case 'ΔR':
                        shift = value*lsqpars.pop(0)
                        spl = splrep(R-shift, VTd[indx])
                        VTd[indx] = splev(R, spl)
                        if self.verbose:
                            print(f'ΔR: {shift:.3f}')

                    case 'ΔV':
                        shift = value*lsqpars.pop(0)
                        VTd[indx] += shift
                        if self.verbose:
                            print(f'ΔV: {shift:.3f}')

                    case 'Vstr':
                        scaling = value*lsqpars.pop(0)
                        Voo = VTd[indx][-1]  # dissociation limit
                        VTd[indx] = (VTd[indx] - Voo)*scaling + Voo
                        if self.verbose:
                            print(f'Vstr: {scaling:.3f}')

                    # multi-value parameters
                    case 'Rstr':
                        inner = value['inner']*lsqpars.pop(0)
                        outer = value['outer']*lsqpars.pop(0)
                        Rscaled = R.copy()
                        Re = R[VTd[indx].argmin()]
                        if self.verbose:
                            print(f'Rstr: inner={inner:.3f}, outer={outer:.3f}')

                        ri = R < Re
                        Rscaled[ri] = (R[ri] - Re)*inner + Re
                        ro = R >= Re
                        Rscaled[ro] = (R[ro] - Re)*outer + Re

                        spl = splrep(Rscaled, VTd[indx])
                        VTd[indx] = splev(R, spl)

                    case 'Wei' | 'Julienne':
                        analyt_dict = state_dict[param].copy()

                        for k, v in state_dict[param].items():
                            if k not in ['Rm', 'Rn']:
                                # reassign analytical PEC parameter in dict
                                analyt_dict[k] = v*lsqpars.pop(0)

                        subR = np.logical_and(R >= value['Rm'],
                                              R <= value['Rn'])

                        VTd[indx][subR] = eval(param)(R[subR], **analyt_dict)

                    case 'spline':
                        spl = splrep(value, lsqpars[:len(value)])

                        subR = np.logical_and(R >= value[0], R <= value[-1])
                        VTd[indx][subR] *= splev(R[subR], spl)

                        lsqpars = lsqpars[len(value):]

            # return modified PECs to csemodel
            self.csemodel.us.VT[indx, indx] = VTd[indx]/self._evcm  # diagonal

        # coupling -----------------------------------------
        for lbl, scaling in self.coup_adj.items():
            scaling = lsqpars.pop(0)
            coupling, i, j = self.coupling[lbl]  # original value tuple
            self.csemodel.us.VT[i][j] = self.csemodel.us.VT[j][i]\
                                      = coupling*scaling
            if self.verbose:
                print(f'coupling {i}{j} = '
                      f'{(coupling*self._evcm*scaling).max():8.2f}')

        # etdm -------------------------------------------------
        for lbl, scaling in self.etdm_adj.items():
            scaling = lsqpars.pop(0)
            etdm, indxi, indxf = self.etdm[lbl]  # original value tuple
            self.csemodel.dipolemoment[:, indxi, indxf] = etdm*scaling


    def __repr__(self):
        lsqpars = list(self.result.x)
        stderr = list(self.result.stderr)

        unit = {'Wei': {'Re':'Å', 'De':'cm⁻¹', 'Voo':'cm⁻¹', 'b':'', 'h':''},
                'Julienne': {'Mx':'cm⁻¹/Å', 'Rx':'Å', 'Vx':'cm⁻¹',
                             'Voo':'cm⁻¹'},
                'Rstr':{'inner':'', 'outer':''}
               }

        about = '\n\nModel fitted parameters\n'

        if self.VT_adj:
            about += 'Potential energy curves ----------------------------\n'
            for state, state_dict in self.VT_adj.items():
                about += f'{state:10s}\n'
                for param, value in state_dict.items():
                    about += f'{param:5s}\n'
                    match param:
                        # single value
                        case 'ΔR' | 'ΔV' | 'Vstr':
                            scaling = lsqpars.pop(0)
                            scal_err = float(stderr.pop(0))
                            about += ' scaling    value*scaling\n'
                            about += f' {scaling:5.3f}±{scal_err:.3f}' +\
                                     f' {value*scaling:8.3f}±' +\
                                     f'{value*scal_err:.3f} cm⁻¹\n'
                        # multi-value
                        case 'Wei' | 'Julienne' | 'Rstr':
                            for k, v in value.items():
                                if k in ['Rm', 'Rn']:
                                    continue
                                scaling = lsqpars.pop(0)
                                scal_err = stderr.pop(0)
                                about += f'{" ":7s} ' +\
                                      f'{scaling:5.3f}±{scal_err:.3f}  ' +\
                                      f'{k:>5s} = {v*scaling:12.3f}±' +\
                                      f'{v*scal_err:.3f} ' +\
                                      f'{unit[param][k]}\n'
                        case 'spline':
                            about += f'  r(Å)         scaling\n'
                            for r in value:  # here is an array
                                about += f'{" ":16s} {r:8.3f}  ' +\
                                      f'{lsqpars.pop(0):12.3f}±' +\
                                      f'{stderr.pop(0):.3f}\n'
                about += '\n'

        if self.coup_adj:
            about += '\nCoupling ------------------------------------\n'
            for lbl, v in self.coup_adj.items():
                scaling = lsqpars.pop(0)
                scaling_err = stderr.pop(0)

                coupling, i, j = self.coupling[lbl]
                ix = np.abs(self.csemodel.us.VT[i, i] -
                            self.csemodel.us.VT[j, j]).argmin()
                Rx = self.R[ix]

                about += f'{lbl:>10s} {" ":14s} {scaling:8.3f}±' +\
                         f'{scaling_err:.3f}'
                about += f'  {coupling[ix]*scaling*self._evcm:8.3f}±' +\
                         f'{coupling[ix]*scaling_err*self._evcm:.3f}' +\
                         f' cm⁻¹ at {Rx:5.3f} Å\n'

        if self.etdm_adj:
            about += '\nElectronic transition dipole moment ---------\n'
            ire = self.csemodel.us.VT[0, 0].argmin()
            Re = self.R[ire]
            for lbl, v in self.etdm_adj.items():
                scaling = lsqpars.pop(0)
                scaling_err = stderr.pop(0)
                etdm, i, f = self.etdm[lbl]
                etdm = etdm[ire]
                about += f'{lbl:>10s} {" ":15s} ' +\
                         f'{scaling:8.3f}±{scaling_err:.3f}' +\
                         f' {etdm*scaling:8.3f}±{etdm*scaling_err:.3f} a.u.' +\
                         f' at {Re:5.3f} Å\n'

        about += '\n'
        return about


    def cross_section(self, data, channel='total', eni=1000, roti=0, rotf=0):
        if data[0][0] < 100:
            dwn = 1000
            wavenumber = []
            self.peak = np.zeros_like(data[1])
            for p in data[1]:
                wavenumber.append(np.arange(p-dwn, p+dwn, 5))
            wavenumber = np.ravel(wavenumber)
        else:
            wavenumber = data[0]

        self.csemodel.calculate_xs(transition_energy=wavenumber,
                                   eni=eni, rotf=rotf, roti=roti)

        if channel == 'total':
            self.csexs = self.csemodel.xs.sum(axis=1)
        else:
            chnl_indx = self.csemodel.us.statelabel.index(channel)
            self.csexs = self.csemodel.xs[:, chnl_indx]

        if data[0][0] < 100:
            # peak position for each input value
            for i, p in enumerate(data[1]):
                subr = np.logical_and(wavenumber > p-dwn, wavenumber < p+dwn)
                self.peak[i] = wavenumber[subr][self.csexs[subr].argmax()]
            

    # least-squares fit -----------------------------------------------------
    def residual(self, pars):
        self.parameter_unpack(pars)  # resets csemodel with modified parameters

        self.diff = []

        for channel, data_dict in self.data2fit.items():
            for data_type, data in data_dict.items():
                wgt = data[-1]
                match data_type[:2]:
                    case 'xs':
                        self.cross_section(data[:2], channel)
                        diff = (self.csexs - data[1])*1e19*wgt
                        self.diff.append(diff)

                    case 'po':
                        if self.csemodel.us.limits[1] == 1:  # single PEC
                            # ensure iterable 
                            if not hasattr(data[0], '__iter__'):
                                data = ([data[0]], [data[1]], [wgt])

                            for v, Tv in zip(*data[:2]):
                                self.csemodel.us.solve(Tv)
                                self.diff.append((self.csemodel.us.cm - Tv)*wgt)
                        else:
                            self.cross_section(data, channel)
                            self.diff.append((self.peak - data[1])*wgt)

                        if self.verbose:
                            print('Δposition: ', self.diff[-len(data[0]):])

                    case 'Bv':
                        if self.csemodel.us.limits[1] == 1:  # single PEC
                            self.csemodel.us.levels(data[0].max()+2)
                            # ensure iterable 
                            if not hasattr(data[0], '__iter__'):
                                data = ([data[0]], [data[1]], [wgt])

                            for v, B in zip(*data[:2]):
                                self.diff.append(
                                   (self.csemodel.us.results[v][1] - B)*wgt)
                        else:
                            self.cross_section(data, channel)
                            self.peak0 = self.peak
                            self.cross_section(data, channel, roti=10, rotf=10)
                            Bv = np.abs(self.peak0 - self.peak)/10/11
                            self.diff.append((Bv - data[1])*wgt)

                        if self.verbose:
                            print('ΔBv: ', self.diff[-len(data[0]):])

        self.diff = np.hstack((self.diff))
        self.sum = self.diff.sum()

        if self.verbose:
            print(f'{pars} {self.sum:g}')

        return self.diff

    def fit(self):

        self.parameter_pack()

        if self.method == 'lm':
            self.result = least_squares(self.residual, self.lsqpars,
                                        method=self.method, 
                                        x_scale='jac', diff_step=0.1)
        else:
            self.result = least_squares(self.residual, self.lsqpars,
                                        method=self.method, bounds=self.bounds,
                                        x_scale='jac', diff_step=0.1)
