import numpy as np

from neat import NeuronSimTree, GreensTree

import utils, data, optimizer, nmda
from matplotlibsettings import *


# parameters g_h distribution
# PVALS_1 = [1000., 1000., 5000., 10000., -40., 0.]
# PVALS_2 = [   0.,    0., 5000., 10000., -40., 0.]
PVALS_1 = [1000., 1000., 6000., 6000., -40., 0.]
PVALS_2 = [   0.,    0., 6000., 6000., -40., 0.]

# apical synapse sites
AP_LOCS = [(47, .5), (38, .9)]

# base synaptic conductances
G_MAX_AMPA = 0.0001  # uS, 100 pS
G_MAX_NMDA = 2000.   # pS, 2000 pS

P_COLORS = ['orange', 'navy', 'darkturquoise']


class HCurrentAnalyzer():
    def __init__(self):

        channel_names = ['L', 'K_ir', 'K_m35']

        # reduced trees
        full_tree, red_tree, self.full_locs, self.red_locs = data.reduceMorphology()
        sim_tree = red_tree.__copy__(new_tree=NeuronSimTree())
        # measured data
        self.v_dat = data.DataContainer(with_zd=True)
        # get file name
        file_name = utils.getFileName(channel_names, True)

        print(file_name)
        # load hall of fame
        with open(file_name, 'rb') as file:
            hall_of_fame = pickle.load(file)
        # get the original simtree with the potassium params
        self.model_evaluator = optimizer.ModelEvaluator(sim_tree, self.v_dat,
                                self.red_locs[0], self.red_locs[1],
                                channel_names=channel_names, mode='evaluate')
        self.model_evaluator.setParameterValues(hall_of_fame[0])

        self._createHcnTees()
        self._createGreensTrees()

    def _createHcnTees(self):
        """
        _1 -> decrease in s2d attenuation in control compared to zd
        _2 -> increase in s2d attenuation in control compared to zd
        """
        global PVALS_1, PVALS_2

        sim_tree = self.model_evaluator.getTreeWithParams()

        self.att_evaluator_1 = optimizer.AttenuationEvaluator(sim_tree, 0.9, 0.8, self.red_locs[0], self.red_locs[1])
        self.att_evaluator_2 = optimizer.AttenuationEvaluator(sim_tree, 1.1, 0.8, self.red_locs[0], self.red_locs[1])

        self.att_evaluator_1.setParameterValues(PVALS_1)
        self.att_evaluator_2.setParameterValues(PVALS_2)

    def _createGreensTrees(self):
        """
        _1 -> decrease in s2d attenuation in control compared to zd
        _2 -> increase in s2d attenuation in control compared to zd
        """
        sim_tree_zd   = self.model_evaluator.getTreeWithParams()
        sim_tree_wh_1 = self.att_evaluator_1.getTreeWithParams()
        sim_tree_wh_2 = self.att_evaluator_2.getTreeWithParams()

        self.greens_tree_zd   = sim_tree_zd.__copy__(new_tree=GreensTree())
        self.greens_tree_wh_1 = sim_tree_wh_1.__copy__(new_tree=GreensTree())
        self.greens_tree_wh_2 = sim_tree_wh_2.__copy__(new_tree=GreensTree())

        # for n1, n2, n3 in zip(self.greens_tree_zd, self.greens_tree_wh_1, self.greens_tree_wh_2): print(n1.e_eq, n2.e_eq, n3.e_eq)

        # self.greens_tree_zd.setEEq(-60.)
        # self.greens_tree_wh_1.setEEq(-60.)
        # self.greens_tree_wh_2.setEEq(-60.)

        self.greens_tree_zd.setCompTree()
        self.greens_tree_wh_1.setCompTree()
        self.greens_tree_wh_2.setCompTree()

        self.greens_tree_zd.setImpedance(np.array([0.]))
        self.greens_tree_wh_1.setImpedance(np.array([0.]))
        self.greens_tree_wh_2.setImpedance(np.array([0.]))

    def plotAttenuation(self, axes=None, units='pS/um2'):
        """
        _1 -> decrease in s2d attenuation in control compared to zd
        _2 -> increase in s2d attenuation in control compared to zd
        """
        if units == 'uS/cm2' :
            unit_conv = 1
            unit_str = r'$\mu$S/cm$^2$'
        elif units == 'pS/um2':
            unit_conv = 0.01
            unit_str = r'pS/um$^2$'
        else:
            print('Unit type not defined, reverting to uS/cm2')
            unit_conv = 1
            units = 'uS/cm2'

        if axes is None:
            pl.figure('g hcn')
            ax0 = myAx(pl.subplot(221))
            ax1 = myAx(pl.subplot(223))

            axa = myAx(pl.subplot(222))
            axb = myAx(pl.subplot(224))

        else:
            ax0 = axes[0]
            ax1 = myAx(axes[1])

            axa = myAx(axes[2])
            axb = myAx(axes[3])

        res_zd = self.model_evaluator.runSim()
        res_1  = self.att_evaluator_1.runSim()
        res_2  = self.att_evaluator_2.runSim()

        # compute attenuation with zd
        att_feature = data.ATTFeature(self.v_dat)
        att_d2s_zd, att_s2d_zd = att_feature.calcAttenuation(res_zd['v_m'][0][:-1], res_zd['v_m'][1][:-1])

        # compute attenuation for two hcn channels densities
        att_d2s_wh_1, att_s2d_wh_1 = att_feature.calcAttenuation(res_1['v_m'][0][:-1], res_1['v_m'][1][:-1])
        att_d2s_wh_2, att_s2d_wh_2 = att_feature.calcAttenuation(res_2['v_m'][0][:-1], res_2['v_m'][1][:-1])

        xvals = np.linspace(0., 750., 1000)

        p_tree = self.model_evaluator.getTreeWithParams()
        p_tree.plot2DMorphology(ax0, use_radius=False, draw_soma_circle=False, lims_margin=0.,
                                plotargs={'lw': lwidth/1.3, 'c': 'DarkGrey'},
                                marklocs=self.red_locs,
                                locargs=[{'marker': 's', 'ms': markersize, 'mfc': 'k', 'mec': 'k'},
                                         {'marker': 's', 'ms': markersize, 'mfc': 'k', 'mec': 'k'}])


        ax1.plot(xvals, np.zeros_like(xvals), lw=lwidth, c='orange', label='with ZD')
        ax1.plot(xvals, [self.att_evaluator_2._h_distr_func(x)*unit_conv for x in xvals],
                                                    lw=lwidth, c='navy', label='Hcn 1')
        ax1.plot(xvals, [self.att_evaluator_1._h_distr_func(x)*unit_conv for x in xvals],
                                                    lw=lwidth, c='darkturquoise', label='Hcn 2')
        ax1.axvline(p_tree.pathLength(self.red_locs[0], self.red_locs[1]), c='k', lw=lwidth*.7, ls='--')

        ax1.set_xlabel(r'$d_{soma}$ ($\mu$m)')

        ax1.set_ylabel(r'$\overline{g}_{Hcn}$ ('+unit_str+')')

        myLegend(ax1, loc='lower center', bbox_to_anchor=(.5,1.0), fontsize=ticksize,
                        handlelength=.5, handletextpad=.2, ncol=3, columnspacing=.4)

        xwidth = 1./6.
        xvals = np.array([0., 1.])

        print(xvals - xwidth, )

        axa.bar(xvals-xwidth, [att_d2s_zd[0]  , att_d2s_zd[1]]  , width=xwidth, align='center', linewidth=lwidth*0.6, edgecolor='k', color='orange', label='with ZD')
        axa.bar(xvals       , [att_d2s_wh_2[0], att_d2s_wh_2[1]], width=xwidth, align='center', linewidth=lwidth*0.6, edgecolor='k', color='navy', label='Hcn 1')
        axa.bar(xvals+xwidth, [att_d2s_wh_1[0], att_d2s_wh_1[1]], width=xwidth, align='center', linewidth=lwidth*0.6, edgecolor='k', color='darkturquoise', label='Hcn 2')

        axb.bar(xvals-xwidth, [att_s2d_zd[0]  , att_s2d_zd[1]]  , width=xwidth, align='center', linewidth=lwidth*0.6, edgecolor='k', color='orange')
        axb.bar(xvals       , [att_s2d_wh_2[0], att_s2d_wh_2[1]], width=xwidth, align='center', linewidth=lwidth*0.6, edgecolor='k', color='navy')
        axb.bar(xvals+xwidth, [att_s2d_wh_1[0], att_s2d_wh_1[1]], width=xwidth, align='center', linewidth=lwidth*0.6, edgecolor='k', color='darkturquoise')

        axa.set_xticks(xvals)
        axa.set_xticklabels([])

        axb.set_xticks(xvals)
        axb.set_xticklabels([-300, 100])
        axb.set_xlabel(r'$I_{inj}$ (pA)')

        axa.set_ylabel(r'Att $D \rightarrow S$')
        axb.set_ylabel(r'Att $S \rightarrow D$')

        axa.set_ylim((0.5, 1.0))
        axb.set_ylim((0.5, 1.0))

        axa.set_yticks([0.6,0.8])
        axb.set_yticks([0.6,0.8])

        myLegend(axa, loc='center left', bbox_to_anchor=(1.0,0.5), fontsize=ticksize,
                        handlelength=.5, handletextpad=.2)

        if axes is None:
            pl.tight_layout()
            pl.show()

    def plotLocalNMDA(self, axes=None, n_syns=np.arange(1,4), resc_flag=True, w_morph=True):
        global AP_LOCS, G_MAX_AMPA, G_MAX_NMDA
        global P_COLORS

        idx = np.argmin(np.abs(n_syns - 18))

        sim_tree_zd   = self.model_evaluator.getTreeWithParams()
        sim_tree_wh_1 = self.att_evaluator_1.getTreeWithParams()
        sim_tree_wh_2 = self.att_evaluator_2.getTreeWithParams()

        nmda_tree_zd   = sim_tree_zd.__copy__(new_tree=nmda.NMDASimTree())
        nmda_tree_wh_1 = sim_tree_wh_1.__copy__(new_tree=nmda.NMDASimTree())
        nmda_tree_wh_2 = sim_tree_wh_2.__copy__(new_tree=nmda.NMDASimTree())

        nmda_tree_zd.setCompTree()
        nmda_tree_wh_1.setCompTree()
        nmda_tree_wh_2.setCompTree()

        z_zd   = self.greens_tree_zd.calcImpedanceMatrix([(1,.5)] + AP_LOCS)[0]
        z_wh_1 = self.greens_tree_wh_1.calcImpedanceMatrix([(1,.5)] + AP_LOCS)[0]
        z_wh_2 = self.greens_tree_wh_2.calcImpedanceMatrix([(1,.5)] + AP_LOCS)[0]

        print('z_zd:')
        print(z_zd)
        print('z_wh_1:')
        print(z_wh_1)
        print('z_wh_2:')
        print(z_wh_2)

        z_zd   = self.greens_tree_zd.calcImpedanceMatrix([AP_LOCS[0]])[0,0]
        z_wh_1 = self.greens_tree_wh_1.calcImpedanceMatrix([AP_LOCS[0]])[0,0]
        z_wh_2 = self.greens_tree_wh_2.calcImpedanceMatrix([AP_LOCS[0]])[0,0]

        # nmda activation for zd case
        _, res_nmda_zd, res_sim_zd = nmda_tree_zd.findNMDAThreshold(AP_LOCS[0], n_syns, G_MAX_AMPA, G_MAX_NMDA)

        # nmda activation with h-current without compensation
        _, res_nmda_wh_1, res_sim_wh_1 = nmda_tree_wh_1.findNMDAThreshold(AP_LOCS[0], n_syns, G_MAX_AMPA, G_MAX_NMDA)
        _, res_nmda_wh_2, res_sim_wh_2 = nmda_tree_wh_2.findNMDAThreshold(AP_LOCS[0], n_syns, G_MAX_AMPA, G_MAX_NMDA)

        # nmda activation with h-current with compensation
        f_wh_1 = z_zd / z_wh_1 * 0.93
        f_wh_2 = z_zd / z_wh_2 * 0.90
        _, resc_nmda_wh_1, resc_sim_wh_1 = nmda_tree_wh_1.findNMDAThreshold(AP_LOCS[0], n_syns, f_wh_1*G_MAX_AMPA, f_wh_1*G_MAX_NMDA)
        _, resc_nmda_wh_2, resc_sim_wh_2 = nmda_tree_wh_2.findNMDAThreshold(AP_LOCS[0], n_syns, f_wh_2*G_MAX_AMPA, f_wh_2*G_MAX_NMDA)

        # save nmda activation curves
        with open(paths.tool_path + 'hcn_att_nmda_act.p', 'wb') as f:
            dill.dump(res_nmda_zd, f)
            dill.dump(res_nmda_wh_1, f)
            dill.dump(res_nmda_wh_2, f)

        if axes is None:
            pl.figure('morph')
            axm = pl.subplot(211)
            axm_ = pl.subplot(212)

            pl.figure('v dend no resc')
            ax0 = pl.subplot(131)
            ax1 = pl.subplot(132)
            ax2 = pl.subplot(133)

            # pl.figure('v dend w resc')
            # axr0 = pl.subplot(131)
            # axr1 = pl.subplot(132)
            # axr2 = pl.subplot(133)

            pl.figure('amp nmda no resc')
            axa = myAx(pl.subplot(211))
            axb = myAx(pl.subplot(212))

            # pl.figure('amp nmda w resc')
            # axra = myAx(pl.subplot(211))
            # axrb = myAx(pl.subplot(212))

            pl.figure('v soma no resc')
            ax0_ = pl.subplot(131)
            ax1_ = pl.subplot(132)
            ax2_ = pl.subplot(133)

            # pl.figure('v soma w resc')
            # axr0_ = pl.subplot(131)
            # axr1_ = pl.subplot(132)
            # axr2_ = pl.subplot(133)

            pl.figure('amp rel no resc')
            ax0r = myAx(pl.subplot(211))
            ax1r = myAx(pl.subplot(212))

            pl.figure('amp rel w resc')
            axr0r = myAx(pl.subplot(211))
            axr1r = myAx(pl.subplot(212))

        else:
            axm = axes['morph'][0]
            axm_ = axes['morph'][1]

            ax0 = axes['psp'][0]
            ax1 = axes['psp'][1]
            ax2 = axes['psp'][2]

            # axr0 = axes['pspr'][0]
            # axr1 = axes['pspr'][1]
            # axr2 = axes['pspr'][2]

            axa = myAx(axes['act'][0])
            axb = myAx(axes['act'][1])

            # axra = myAx(axes['actr'][0])
            # axrb = myAx(axes['actr'][1])

            ax0_ = axes['psp_'][0]
            ax1_ = axes['psp_'][1]
            ax2_ = axes['psp_'][2]

            # axr0_ = axes['pspr_'][0]
            # axr1_ = axes['pspr_'][1]
            # axr2_ = axes['pspr_'][2]

            ax0r = myAx(axes['amp'][0])
            ax1r = myAx(axes['amp'][1])

            axr0r = myAx(axes['ampr'][0])
            axr1r = myAx(axes['ampr'][1])

        # plot morphology 1
        p_tree = self.model_evaluator.getTreeWithParams()
        p_tree.plot2DMorphology(axm, use_radius=False, draw_soma_circle=False, lims_margin=0.,
                                plotargs={'lw': lwidth/1.3, 'c': 'DarkGrey'},
                                marklocs=[AP_LOCS[0]],
                                locargs=[{'marker': 's', 'ms': markersize, 'mfc': 'k', 'mec': 'k'}])
        # plot morphology 2
        p_tree = self.model_evaluator.getTreeWithParams()
        p_tree.plot2DMorphology(axm_, use_radius=False, draw_soma_circle=False, lims_margin=0.,
                                plotargs={'lw': lwidth/1.3, 'c': 'DarkGrey'},
                                marklocs=[(1,.5), AP_LOCS[0]],
                                locargs=[{'marker': 's', 'ms': markersize, 'mfc': 'k', 'mec': 'k'},
                                         {'marker': 's', 'ms': markersize, 'mfc': 'k', 'mec': 'k'}])

        # plot dendritic voltages
        res = res_sim_zd[idx]
        ax0.plot(res['t'], res['v_m'][1]-res['v_m'][1,0], lw=lwidth, color='orange')

        res = res_sim_wh_2[idx]
        ax1.plot(res['t'], res['v_m'][1]-res['v_m'][1,0], lw=lwidth, color='navy')

        res = res_sim_wh_1[idx]
        ax2.plot(res['t'], res['v_m'][1]-res['v_m'][1,0], lw=lwidth, color='darkturquoise')

        print('\n>>> V_eq')
        print('  > V_eq w zd = %.2f'%res_sim_zd[idx]['v_m'][1,0])
        print('  > V_eq hcn1 = %.2f'%res_sim_wh_2[idx]['v_m'][1,0])
        print('  > V_eq hcn2 = %.2f'%res_sim_wh_1[idx]['v_m'][1,0])

        ax0.set_ylim((-5.,65.))
        ax1.set_ylim((-5.,65.))
        ax2.set_ylim((-5.,65.))

        drawScaleBars(ax0)
        drawScaleBars(ax1)
        drawScaleBars(ax2, xlabel='ms', ylabel='mV', fstr_xlabel=r'%.0f ', fstr_ylabel=r'%.0f ')

        ax0.set_yticks([0.])
        ax0.set_yticklabels([r'$v_{eq}$'])

        ax0.set_title('with ZD', fontsize=ticksize)
        ax1.set_title('Hcn 1', fontsize=ticksize)
        ax2.set_title('Hcn 2', fontsize=ticksize)

        # # plot dendritic voltages with rescale
        # res = res_sim_zd[idx]
        # axr0.plot(res['t'], res['v_m'][1]-res['v_m'][1,0], lw=lwidth, color='orange')

        # res = resc_sim_wh_2[idx]
        # axr1.plot(res['t'], res['v_m'][1]-res['v_m'][1,0], lw=lwidth, color='navy')

        # res = resc_sim_wh_1[idx]
        # axr2.plot(res['t'], res['v_m'][1]-res['v_m'][1,0], lw=lwidth, color='darkturquoise')

        # axr0.set_ylim((-5.,65.))
        # axr1.set_ylim((-5.,65.))
        # axr2.set_ylim((-5.,65.))

        # drawScaleBars(axr0)
        # drawScaleBars(axr1)
        # drawScaleBars(axr2, xlabel='ms', ylabel='mV', fstr_xlabel=r'%.0f ', fstr_ylabel=r'%.0f ')

        # axr0.set_yticks([0.])
        # axr0.set_yticklabels([r'$v_{eq}$'])

        # axr0.set_title('with ZD', fontsize=labelsize)
        # axr1.set_title('Hcn 1', fontsize=labelsize)
        # axr2.set_title('Hcn 2', fontsize=labelsize)

        # plot nmda amplitudes without rescale
        axa.plot(n_syns, res_nmda_zd['amp'], lw=lwidth, color='orange', label='with ZD')
        axa.plot(n_syns, res_nmda_wh_2['amp'], lw=lwidth, color='navy', label='Hcn 1')

        axb.plot(n_syns, res_nmda_zd['amp'], lw=lwidth, color='orange', label='with ZD')
        axb.plot(n_syns, res_nmda_wh_1['amp'], lw=lwidth, color='darkturquoise', label='Hcn 2')

        myLegend(axa, loc='center left', bbox_to_anchor=(1.0,0.5), fontsize=ticksize, handlelength=.9, handletextpad=.2)
        myLegend(axb, loc='center left', bbox_to_anchor=(1.0,0.5), fontsize=ticksize, handlelength=.9, handletextpad=.2)

        axa.set_ylabel(r'$v_{amp}$ (mV)')
        axb.set_ylabel(r'$v_{amp}$ (mV)')
        # axb.set_yticklabels([])

        # axa.set_xlabel(r'$N_{syn}$')
        axb.set_xlabel(r'$N_{syn}$')
        axa.set_xticklabels([])

        # # plot nmda amplitudes with rescale
        # axra.plot(n_syns, res_nmda_zd['amp'], lw=lwidth, color='orange', label='with ZD')
        # axra.plot(n_syns, resc_nmda_wh_2['amp'], lw=lwidth, color='navy', label='Hcn 1')

        # axrb.plot(n_syns, res_nmda_zd['amp'], lw=lwidth, color='orange', label='with ZD')
        # axrb.plot(n_syns, resc_nmda_wh_1['amp'], lw=lwidth, color='darkturquoise', label='Hcn 2')

        # myLegend(axra, loc='center left', bbox_to_anchor=(1.0,0.5), fontsize=ticksize, handlelength=.9, handletextpad=.2)
        # myLegend(axrb, loc='center left', bbox_to_anchor=(1.0,0.5), fontsize=ticksize, handlelength=.9, handletextpad=.2)

        # axra.set_ylabel(r'$v_{amp}$ (mV)')
        # # axrb.set_ylabel(r'$v_{amp}$ (mV)')
        # axrb.set_yticklabels([])

        # axra.set_xlabel(r'$N_{syn}$')
        # axrb.set_xlabel(r'$N_{syn}$')
        # # axra.set_xticklabels([])

        # plot somatic voltages without rescale
        res = res_sim_zd[idx]
        ax0_.plot(res['t'], res['v_m'][0]-res['v_m'][0,0], lw=lwidth, color='orange')

        res = res_sim_wh_2[idx]
        ax1_.plot(res['t'], res['v_m'][0]-res['v_m'][0,0], lw=lwidth, color='navy')

        res = res_sim_wh_1[idx]
        ax2_.plot(res['t'], res['v_m'][0]-res['v_m'][0,0], lw=lwidth, color='darkturquoise')

        ax0_.set_ylim((-.5,7.))
        ax1_.set_ylim((-.5,7.))
        ax2_.set_ylim((-.5,7.))

        drawScaleBars(ax0_)
        drawScaleBars(ax1_)
        drawScaleBars(ax2_, xlabel='ms', ylabel='mV', fstr_xlabel=r'%.0f ', fstr_ylabel=r'%.0f ')

        ax0_.set_yticks([0.])
        ax0_.set_yticklabels([r'$v_{eq}$'])

        ax0_.set_title('with ZD', fontsize=ticksize)
        ax1_.set_title('Hcn 1', fontsize=ticksize)
        ax2_.set_title('Hcn 2', fontsize=ticksize)

        # # plot somatic voltages with rescale
        # res = res_sim_zd[idx]
        # axr0_.plot(res['t'], res['v_m'][0]-res['v_m'][0,0], lw=lwidth, color='orange')

        # res = resc_sim_wh_2[idx]
        # axr1_.plot(res['t'], res['v_m'][0]-res['v_m'][0,0], lw=lwidth, color='navy')

        # res = resc_sim_wh_1[idx]
        # axr2_.plot(res['t'], res['v_m'][0]-res['v_m'][0,0], lw=lwidth, color='darkturquoise')

        # axr0_.set_ylim((-.5,7.))
        # axr1_.set_ylim((-.5,7.))
        # axr2_.set_ylim((-.5,7.))

        # drawScaleBars(axr0_)
        # drawScaleBars(axr1_)
        # drawScaleBars(axr2_, xlabel='ms', ylabel='mV', fstr_xlabel=r'%.0f ', fstr_ylabel=r'%.0f ')

        # axr0_.set_yticks([0.])
        # axr0_.set_yticklabels([r'$v_{eq}$'])

        # axr0_.set_title('with ZD', fontsize=labelsize)
        # axr1_.set_title('Hcn 1', fontsize=labelsize)
        # axr2_.set_title('Hcn 2', fontsize=labelsize)

        # plot relative PSP amplitudes without rescale
        v_dend_list = [res_sim_zd[idx]['v_m'][1], res_sim_wh_2[idx]['v_m'][1], res_sim_wh_1[idx]['v_m'][1]]
        v_soma_list = [res_sim_zd[idx]['v_m'][0], res_sim_wh_2[idx]['v_m'][0], res_sim_wh_1[idx]['v_m'][0]]

        v_amp_dend = [nmda.calcAmpWidthSurface(v_arr, nmda.T0+nmda.DELTA_T)[0] for v_arr in v_dend_list]
        v_amp_soma = [nmda.calcAmpWidthSurface(v_arr, nmda.T0+nmda.DELTA_T)[0] for v_arr in v_soma_list]

        print('v amp abs')
        print(v_amp_dend)
        print(v_amp_soma)

        v_amp_dend = np.array(v_amp_dend) / v_amp_dend[0]
        v_amp_soma = np.array(v_amp_soma) / v_amp_soma[0]

        print('v amp rel')
        print(v_amp_dend)
        print(v_amp_soma)

        plabels = [r'with ZD', r'Hcn 1', r'Hcn 2']

        xvals = [1.,2.,3.]

        for ii, (xv, v_amp) in enumerate(zip(xvals, v_amp_dend)):
            ax0r.plot([xv], v_amp, marker=mfs[ii], mfc=P_COLORS[ii], mec='k', ms=markersize, label=plabels[ii])
        ax0r.axhline(1., lw=lwidth*.6, ls='--', c='k')

        for ii, (xv, v_amp) in enumerate(zip(xvals, v_amp_soma)):
            ax1r.plot([xv], v_amp, marker=mfs[ii], mfc=P_COLORS[ii], mec='k', ms=markersize, label=plabels[ii])
        ax1r.axhline(1., lw=lwidth*.6, ls='--', c='k')

        ax0r.set_ylim((0.8, 1.2))
        ax1r.set_ylim((0.8, 1.2))

        ax0r.set_yticks([0.9,1.0,1.1])
        ax1r.set_yticks([0.9,1.0,1.1])

        ax0r.set_xlim((.7,3.3))
        ax1r.set_xlim((.7,3.3))

        ax0r.set_ylabel(r'$v_{amp} / v_{amp}^{ZD}$')
        ax1r.set_ylabel(r'$v_{amp} / v_{amp}^{ZD}$')
        # ax1r.set_yticklabels([])

        ax0r.set_xticks(xvals)
        # ax0r.set_xticklabels(plabels, rotation=45)
        ax0r.set_xticklabels([])

        ax1r.set_xticks(xvals)
        ax1r.set_xticklabels(plabels, rotation=45)

        ax0r.annotate(r'Dend', (.5,.8), xycoords='axes fraction', fontsize=labelsize, ha='center', va='center')
        ax1r.annotate(r'Soma', (.5,.8), xycoords='axes fraction', fontsize=labelsize, ha='center', va='center')

        # plot relative PSP amplitudes with rescale
        v_dend_list = [res_sim_zd[idx]['v_m'][1], resc_sim_wh_2[idx]['v_m'][1], resc_sim_wh_1[idx]['v_m'][1]]
        v_soma_list = [res_sim_zd[idx]['v_m'][0], resc_sim_wh_2[idx]['v_m'][0], resc_sim_wh_1[idx]['v_m'][0]]

        v_amp_dend = [nmda.calcAmpWidthSurface(v_arr, nmda.T0+nmda.DELTA_T)[0] for v_arr in v_dend_list]
        v_amp_soma = [nmda.calcAmpWidthSurface(v_arr, nmda.T0+nmda.DELTA_T)[0] for v_arr in v_soma_list]

        print('v amp abs')
        print(v_amp_dend)
        print(v_amp_soma)

        v_amp_dend = np.array(v_amp_dend) / v_amp_dend[0]
        v_amp_soma = np.array(v_amp_soma) / v_amp_soma[0]

        print('v amp rel')
        print(v_amp_dend)
        print(v_amp_soma)

        plabels = [r'with ZD', r'Hcn 1', r'Hcn 2']

        xvals = [1.,2.,3.]

        for ii, (xv, v_amp) in enumerate(zip(xvals, v_amp_dend)):
            axr0r.plot([xv], v_amp, marker=mfs[ii], mfc=P_COLORS[ii], mec='k', ms=markersize, label=plabels[ii])
        axr0r.axhline(1., lw=lwidth*.6, ls='--', c='k')

        for ii, (xv, v_amp) in enumerate(zip(xvals, v_amp_soma)):
            axr1r.plot([xv], v_amp, marker=mfs[ii], mfc=P_COLORS[ii], mec='k', ms=markersize, label=plabels[ii])
        axr1r.axhline(1., lw=lwidth*.6, ls='--', c='k')

        axr0r.set_ylim((0.8, 1.2))
        axr1r.set_ylim((0.8, 1.2))

        axr0r.set_yticks([0.9,1.0,1.1])
        axr1r.set_yticks([0.9,1.0,1.1])

        axr0r.set_xlim((.7,3.3))
        axr1r.set_xlim((.7,3.3))

        axr0r.set_yticklabels([])
        # axr0r.set_ylabel(r'$v_{amp} / v_{amp}^{ZD}$')
        # axr1r.set_ylabel(r'$v_{amp} / v_{amp}^{ZD}$')
        axr1r.set_yticklabels([])

        axr0r.set_xticks(xvals)
        axr0r.set_xticklabels([])
        # axr0r.set_xticklabels(plabels, rotation=45)

        axr1r.set_xticks(xvals)
        axr1r.set_xticklabels(plabels, rotation=45)

        axr0r.set_title(r'$v_{amp}$ dend'+'\nnormalized', fontsize=ticksize, ma='left')

        # axr0r.annotate(r'Dend', (.5,.9), xycoords='axes fraction', fontsize=labelsize, ha='center', va='center')
        # axr1r.annotate(r'Soma', (.5,.9), xycoords='axes fraction', fontsize=labelsize, ha='center', va='center')

        # myLegend(ax0r, loc='center left', bbox_to_anchor=(1.0, 0.5))

        if axes is None:
            pl.tight_layout()
            pl.show()


    def plotNMDAInteraction(self, axes=None, n_syns=np.arange(1,4)):
        global AP_LOCS, G_MAX_AMPA, G_MAX_NMDA
        global P_COLORS

        idx = np.argmin(np.abs(n_syns - 18))
        ns_ = n_syns[idx]

        sim_tree_zd   = self.model_evaluator.getTreeWithParams()
        sim_tree_wh_1 = self.att_evaluator_1.getTreeWithParams()
        sim_tree_wh_2 = self.att_evaluator_2.getTreeWithParams()

        nmda_tree_zd   = sim_tree_zd.__copy__(new_tree=NMDASimTree())
        nmda_tree_wh_1 = sim_tree_wh_1.__copy__(new_tree=NMDASimTree())
        nmda_tree_wh_2 = sim_tree_wh_2.__copy__(new_tree=NMDASimTree())

        nmda_tree_zd.setCompTree()
        nmda_tree_wh_1.setCompTree()
        nmda_tree_wh_2.setCompTree()

        z_zd   = self.greens_tree_zd.calcImpedanceMatrix([(1,.5)] + AP_LOCS)[0]
        z_wh_1 = self.greens_tree_wh_1.calcImpedanceMatrix([(1,.5)] + AP_LOCS)[0]

        z_zd   = self.greens_tree_zd.calcImpedanceMatrix(AP_LOCS)[0,0,0]
        z_wh_1 = self.greens_tree_wh_1.calcImpedanceMatrix(AP_LOCS)[0,0,0]
        z_wh_2 = self.greens_tree_wh_2.calcImpedanceMatrix(AP_LOCS)[0,0,0]

        # nmda activation for zd case
        _, res_nmda_zd, res_sim_zd = nmda_tree_zd.findNMDAThreshold(AP_LOCS[0], n_syns, G_MAX_AMPA, G_MAX_NMDA, loc_=AP_LOCS[1], n_syn_=2*ns_)

        res_zd_ =  nmda_tree_zd.runExperiment(AP_LOCS[0], G_MAX_AMPA, G_MAX_NMDA, n_syn=0, loc_=AP_LOCS[1], n_syn_=2*ns_)

        # nmda activation with h-current with compensation
        # f_wh_1 = z_zd / z_wh_1
        # f_wh_2 = z_zd / z_wh_2
        f_wh_1 = 1.
        f_wh_2 = 1.
        _, resc_nmda_wh_1, resc_sim_wh_1 = nmda_tree_wh_1.findNMDAThreshold(AP_LOCS[0], n_syns, f_wh_1*G_MAX_AMPA, f_wh_1*G_MAX_NMDA, loc_=AP_LOCS[1], n_syn_=2*ns_)
        _, resc_nmda_wh_2, resc_sim_wh_2 = nmda_tree_wh_2.findNMDAThreshold(AP_LOCS[0], n_syns, f_wh_2*G_MAX_AMPA, f_wh_2*G_MAX_NMDA, loc_=AP_LOCS[1], n_syn_=2*ns_)
        # _, resc_nmda_wh_1, resc_sim_wh_1 = nmda_tree_wh_1.findNMDAThreshold(AP_LOCS[0], n_syns, G_MAX_AMPA, G_MAX_NMDA, loc_=AP_LOCS[1], n_syn_=2*ns_)
        # _, resc_nmda_wh_2, resc_sim_wh_2 = nmda_tree_wh_2.findNMDAThreshold(AP_LOCS[0], n_syns, G_MAX_AMPA, G_MAX_NMDA, loc_=AP_LOCS[1], n_syn_=2*ns_)

        # load nmda activation curves
        with open(paths.tool_path + 'hcn_att_nmda_act.p', 'rb') as f:
            n_syns_ = np.arange(1,25)
            # n_syns_ = np.array([5, 10, 20])
            # n_syns_ = n_syns
            res1_nmda_zd = dill.load(f)
            resc1_nmda_wh_1 = dill.load(f)
            resc1_nmda_wh_2 = dill.load(f)

        if axes is None:
            pl.figure('morph')
            axm = pl.gca()

            pl.figure('v dend')
            ax0 = pl.subplot(131)
            ax1 = pl.subplot(132)
            # ax2 = pl.subplot(133)

            pl.figure('amp nmda')
            axa = myAx(pl.subplot(311))
            axb = myAx(pl.subplot(312))
            axc = myAx(pl.subplot(313))

            pl.figure('thr diff')
            ax_ = myAx(pl.gca())

        else:
            axm = axes['morph']

            ax0 = axes['psp'][0]
            ax1 = axes['psp'][1]

            axa = myAx(axes['act'][0])
            axb = myAx(axes['act'][1])
            axc = myAx(axes['act'][2])

            ax_ = myAx(axes['thr'])

        # plot morphology
        p_tree = self.model_evaluator.getTreeWithParams()
        p_tree.plot2DMorphology(axm, use_radius=False, draw_soma_circle=False, lims_margin=0.,
                                plotargs={'lw': lwidth/1.3, 'c': 'DarkGrey'},
                                marklocs=AP_LOCS,
                                locargs=[{'marker': 's', 'ms': markersize, 'mfc': 'k', 'mec': 'k'},
                                         {'marker': 's', 'ms': markersize, 'mfc': 'k', 'mec': 'k'}])

        # plot dendritic voltages
        res = res_zd_
        ax0.plot(res['t'], res['v_m'][1]-res['v_m'][1,0], lw=lwidth, color='orange')
        res = res_sim_zd[idx]
        ax1.plot(res['t'], res['v_m'][1]-res['v_m'][1,0], lw=lwidth, color='orange')

        ax0.set_ylim((-5.,65.))
        ax1.set_ylim((-5.,65.))

        drawScaleBars(ax0)
        drawScaleBars(ax1, xlabel='ms', ylabel='mV', fstr_xlabel=r'%.0f ', fstr_ylabel=r'%.0f ')

        ax0.set_yticks([0.])
        ax0.set_yticklabels([r'$v_{eq}$'])

        ax0.set_title(r'syn 2', fontsize=labelsize)
        ax1.set_title(r'syn 1 + syn 2', fontsize=labelsize)

        # plot nmda amplitudes
        axa.plot(n_syns_, res1_nmda_zd['amp'], lw=lwidth, color='DarkGrey')
        axa.plot(n_syns, res_nmda_zd['amp'], lw=lwidth, color='orange')

        axb.plot(n_syns_, resc1_nmda_wh_2['amp'], lw=lwidth, color='DarkGrey')
        axb.plot(n_syns, resc_nmda_wh_2['amp'], lw=lwidth, color='navy')

        axc.plot(n_syns_, resc1_nmda_wh_1['amp'], lw=lwidth, color='DarkGrey')
        axc.plot(n_syns, resc_nmda_wh_1['amp'], lw=lwidth, color='darkturquoise')

        # plot thresholds
        axa.axvline(nmda.deviationThrFromLinear(res1_nmda_zd['amp']), c='k', ymax=0.8, lw=lwidth*0.6, ls='--')
        axa.axvline(nmda.deviationThrFromLinear(res_nmda_zd['amp']), c='k', ymax=0.8, lw=lwidth*0.6, ls='--')

        axb.axvline(nmda.deviationThrFromLinear(resc1_nmda_wh_2['amp']), c='k', ymax=0.8,  lw=lwidth*0.6, ls='--')
        axb.axvline(nmda.deviationThrFromLinear(resc_nmda_wh_2['amp']), c='k', ymax=0.8,  lw=lwidth*0.6, ls='--')

        axc.axvline(nmda.deviationThrFromLinear(resc1_nmda_wh_1['amp']), c='k', ymax=0.8,  lw=lwidth*0.6, ls='--')
        axc.axvline(nmda.deviationThrFromLinear(resc_nmda_wh_1['amp']), c='k', ymax=0.8,  lw=lwidth*0.6, ls='--')

        axa.set_ylim((-5.,80.))
        axb.set_ylim((-5.,80.))
        axc.set_ylim((-5.,80.))

        axa.set_xticklabels([])
        axb.set_xticklabels([])

        axc.set_xlabel(r'$N_{syn}$')

        axa.set_ylabel(r'$v_{amp}$ (mV)')
        axb.set_ylabel(r'$v_{amp}$ (mV)')
        axc.set_ylabel(r'$v_{amp}$ (mV)')

        axa.annotate(r'with ZD', (.001,.9), xycoords='axes fraction', fontsize=ticksize, ha='left', va='center')
        axb.annotate(r'Hcn 1', (.001,.9), xycoords='axes fraction', fontsize=ticksize, ha='left', va='center')
        axc.annotate(r'Hcn 2', (.001,.9), xycoords='axes fraction', fontsize=ticksize, ha='left', va='center')

        thr_diff_zd   = nmda.deviationThrFromLinear(res_nmda_zd['amp']) - \
                        nmda.deviationThrFromLinear(res1_nmda_zd['amp'])
        thr_diff_wh_2 = nmda.deviationThrFromLinear(resc_nmda_wh_2['amp']) - \
                        nmda.deviationThrFromLinear(resc1_nmda_wh_2['amp'])
        thr_diff_wh_1 = nmda.deviationThrFromLinear(resc_nmda_wh_1['amp']) - \
                        nmda.deviationThrFromLinear(resc1_nmda_wh_1['amp'])

        thr_diffs = [thr_diff_zd, thr_diff_wh_2, thr_diff_wh_1]
        xvals = [1.,2.,3.]
        plabels = [r'with ZD', r'Hcn 1', r'Hcn 2']

        for ii, (xv, td) in enumerate(zip(xvals, thr_diffs)):
            ax_.plot([xv], td, marker=mfs[ii], mfc=P_COLORS[ii], mec='k', ms=markersize, label=plabels[ii])
        ax_.axhline(0., lw=lwidth*.6, ls='--', c='k')

        ax_.set_xlim((.7,3.3))

        ax_.set_ylabel(r'$\Delta \theta_{NMDA}$ ($N_{syn}$)')

        ax_.set_xticks(xvals)
        ax_.set_xticklabels(plabels, rotation=45)

        if axes is None:
            pl.tight_layout()
            pl.show()


if __name__ == "__main__":
    hca = HCurrentAnalyzer()

    # hca.plotAttenuation()

    # hca.plotLocalNMDA(n_syns=np.array([5, 10, 18]))
    # hca.plotLocalNMDA(n_syns=np.arange(1,25))

    # hca.plotNMDAInteraction(n_syns=np.array([1,10,18]))
    hca.plotNMDAInteraction(n_syns=np.arange(1,25))





