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
        'xs' : {channel statelabel, ([wavenumber], [cross section])}
        'Bv' : {statelabel, ([v], [Bv])}
        'pos' : {statelabel, ([v], [wavenumber of peak])}
        'width' : {statelabel, (wn0, wn1, width, J)}

    Attributes
    ----------
    result : dict
        least_squares fit attriubutes, with stderr for each parameter
    """

    def __init__(self, csemodel, data2fit, VT_adj={}, coup_adj={}, etdm_adj={},
                 verbose=True):
        """
        Parameters
        ----------
        data2fit: {'channel':{'position': wavenumber_of_peak,
                              'Bv: [v, Bv] array,
                              'xs': [wavenumber, cross_section] array},
                  {'channel2':   }}

        # potential energy curves -------
        VT_adj: {PEC:
            {'ΔR':float},  # radial shift
            {'ΔV':float},  # energy shift

            # parameters for Wei analytical PEC, applied between Rm...Rn
            {'Wei':{'re':re, 'De':De, 'voo':voo, 'b':b, 'h':h, 'Rm':Rm,
                    'Rn':Rn}},

            # Julienne (open channel), applied Rm...Rn
            {'Julienne':{'Mx':Mx, 'Rx':Rx, 'Vx':Vx, 'Voo':Voo, 'Rm':Rm,
                         'Rn':Rn}},

            {'spline':[R₀, R₁, ..., R₋₁]},  # radial positions of knots
            PEC is scaled by spline between Rm..Rn 
            )

        # coupling scale factor -------
        coup_adj: {'state0<->state1':scaling, 'state0<->state2':scaling}  # etc.

        # electric dipole transition moment -------
        etdm_adj: {'fsl0<-isl0':scaling_factor, 'fsl1<-isl0':scaling_factor} 

        Each parameter is a scaled relative to 1.0
        """

        self.verbose = verbose

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

        # least-squares fitting ------------------------------
        self.fit()  # least-squares fit
        self.result.stderr = fiterrors(self.result)  # parameter error estimates
        self.print_result()

    def parameter_set(self):
        """ set parameter list for least-squares fitting.

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
        self.bounds = (pars*0.95, pars*1.05)

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
                        spl = splrep(R-value*lsqpars.pop(0), VTd[indx])
                        VTd[indx] = splev(R, spl)

                    case 'ΔV':
                        scaling = lsqpars.pop(0)
                        VTd[indx] += value*scaling

                    # multi-value
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

        # etdm -------------------------------------------------
        for lbl, scaling in self.etdm_adj.items():
            scaling = lsqpars.pop(0)
            etdm, indxi, indxf = self.etdm[lbl]  # original value tuple
            self.csemodel.dipolemoment[:, indxi, indxf] = etdm*scaling


    def print_result(self):
        lsqpars = list(self.result.x)
        stderr = list(self.result.stderr)

        unit = {'Wei': {'re':'Å', 'De':'cm⁻¹', 'voo':'cm⁻¹', 'b':'', 'h':''},
                'Julienne': {'mx':'cm⁻¹/Å', 'rx':'Å', 'vx':'cm⁻¹', 'voo':'cm⁻¹'}
               }

        print('\n\nModel fitted parameters')

        if self.VT_adj:
            print('Potential energy curves ----------------------------')
            for state, state_dict in self.VT_adj.items():
                print(f'{state:10s} ', end='')
                for param, value in state_dict.items():
                    match param:
                        # single value
                        case 'ΔR' | 'ΔV':
                            print(f'{param:15s} '
                                  f'{value*lsqpars.pop(0):8.3f}±'
                                  f'{value*stderr.pop(0):.3f} cm⁻¹') 
                        # multi-value
                        case 'Wei' | 'Julienne':
                            print(f'{param:25s} ')
                            for k, v in value.items():
                                if k in ['Rm', 'Rn']:
                                    continue
                                scaling = lsqpars.pop(0)
                                scaling_err = stderr.pop(0)
                                print(f'{" ":27s} ',
                                      f'{scaling:8.3f}±{scaling_err:.3f}'
                                      f'{k:>5s} = {v*scaling:12.3f}±'
                                      f'{v*scaling_err:.3f} '
                                      f'{unit[param][k]}')
                        case 'spline':
                            print(f'{param:6s}   r(Å)         scaling')
                            for r in value:  # here an array
                                print(f'{" ":16s} {r:8.3f}  '
                                      f'{lsqpars.pop(0):12.3f}±'
                                      f'{stderr.pop(0):.3f}')
                print()

        if self.coup_adj:
            print('\nCoupling ------------------------------------')
            for lbl, v in self.coup_adj.items():
                scaling = lsqpars.pop(0)
                scaling_err = stderr.pop(0)

                coupling, i, j = self.coupling[lbl]
                ix = np.abs(self.csemodel.us.VT[i, i] -
                            self.csemodel.us.VT[j, j]).argmin()
                Rx = self.R[ix]

                print(f'{lbl:>10s} {" ":14s} {scaling:8.3f}±{scaling_err:.3f}',
                      end='')
                print(f'  {coupling[ix]*scaling*self._evcm:8.3f}±'
                      f'{coupling[ix]*scaling_err*self._evcm:.3f}'
                      f' cm⁻¹ at {Rx:5.3f} Å')

        if self.etdm_adj:
            print('\nElectronic transition dipole moment ---------')
            ire = self.csemodel.us.VT[0, 0].argmin()
            Re = self.R[ire]
            for lbl, v in self.etdm_adj.items():
                scaling = lsqpars.pop(0)
                scaling_err = stderr.pop(0)
                etdm, i, f = self.etdm[lbl]
                etdm = etdm[ire]
                print(f'{lbl:>10s} {" ":15s} '
                      f'{scaling:8.3f}±{scaling_err:.3f}'
                      f' {etdm*scaling:8.3f}±{etdm*scaling_err:.3f} a.u.',
                      f' at {Re:5.3f} Å')

        print()

    # least-squares fit -----------------------------------------------------
    def residual(self, pars, keepxs=False):
        self.parameter_unpack(pars)  # resets csemodel with modified parameters

        if keepxs:
            self.csexs = {}

        self.diff = []
        for channel, data_dict in self.data2fit.items():
            if keepxs:
                self.csexs[channel] = {}
            if channel != 'total':
                chnl_indx = self.csemodel.us.statelabel.index(channel)
            for data_type, data in data_dict.items():
                match data_type[:2]:
                    case 'xs':
                        wavenumber, xs = data
                        self.csemodel.calculate_xs(transition_energy=wavenumber)

                        if channel == 'total':
                            csexs = self.csemodel.xs.sum(axis=1)
                        else:
                            csexs = self.csemodel.xs[:, chnl_indx]

                        diff = (csexs-xs)*1e19
                        self.diff.append(diff)

                    case 'po':
                        pos = data
                        wavenumber = np.arange(pos-100, pos+100, 10)
                        self.csemodel.calculate_xs(transition_energy=wavenumber)

                        if channel == 'total':
                            csexs = self.csemodel.xs.sum(axis=1)
                        else:
                            csexs = self.csemodel.xs[:, chnl_indx]

                        self.peak = wavenumber[csexs.argmax()]
                        self.diff.append(np.array(self.peak-pos))

                if keepxs:
                    self.csexs[channel][data_type] = (wavenumber, csexs)


        self.diff = np.concatenate(self.diff).ravel()
        self.sum = self.diff.sum()

        if self.verbose:
            print(f'{pars} {self.sum:g}')
        else:
            print('.', end='')

        return self.diff

    def fit(self):

        self.parameter_set()

        if not self.verbose:
            print('Model_fit: each "." represents an iteration')

        self.result = least_squares(self.residual, self.lsqpars,
                                    # method='lm',
                                    bounds=self.bounds, method='trf',
                                    x_scale='jac', diff_step=0.1)
