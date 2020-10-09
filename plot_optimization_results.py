import numpy as np

from neat import NeuronSimTree

import pickle

import data, utils, optimizer
from matplotlibsettings import *


PSTRINGS = {'L': '$L$', 'L_c': '$L_x$',
            'K_m': '$K_m$', 'K_m35': '$K_m$', 'K_ir': '$K_{ir}$',
            'h_HAY': '$h$', 'h_u': '$h$',
            'Na_p': '$Na_p$', 'NaP': '$NaP$'}
PCOLORS = {'L': colours[0], 'L_c': colours[0],
           'K_m35': colours[1], 'K_ir': colours[2], 'K_m': colours[5],
           'h_u': colours[3], 'h_HAY': colours[4],
           'Na_p': colours[5], 'NaP': colours[6]}


def plotTrace(channel_names, zd, axes=None, plot_xlabel=True, plot_ylabel=True, plot_legend=True, units='uS/cm2', suffix='_predef'):
    ts = np.array([data.T0, data.T1, data.T2, data.T3]) - 100.
    pcolor = 'Orange' if zd else 'Blue'
    if units == 'uS/cm2' :
        unit_conv = 1
    elif units == 'pS/um2':
        unit_conv = 0.01
    else:
        print('Unit type not defined, reverting to uS/cm2')
        unit_conv = 1

    # reduced trees
    full_tree, red_tree, full_locs, red_locs = data.reduceMorphology()
    sim_tree = red_tree.__copy__(new_tree=NeuronSimTree())
    # measured data
    v_dat = data.DataContainer(with_zd=zd)
    # get file name
    file_name = utils.getFileName(channel_names, zd, suffix=suffix)
    # model
    model_evaluator = optimizer.ModelEvaluator(sim_tree, v_dat,
                                                red_locs[0], red_locs[1],
                                                channel_names=channel_names)
    # load hall of fame
    file = open(file_name, 'rb')
    hall_of_fame = pickle.load(file)
    file.close()
    # run the best fit model
    model_evaluator.setParameterValues(hall_of_fame[0])
    print("\n--> running model for parameter values: \n%s"%model_evaluator.toStrParameterValues())
    res = model_evaluator.runSim()

    if axes is None:
        pl.figure('v opt', figsize=(16,4))
        axes = [pl.subplot(161), pl.subplot(162), pl.subplot(163), pl.subplot(164), pl.subplot(165), pl.subplot(166)]
        pshow = True
    else:
        pshow = False

    i0s = [int(tt / data.DT) for tt in ts]
    i1s = [int((tt+1000.-data.DT)/data.DT) for tt in ts]
    t_plot = np.arange(0., 1000.-3*data.DT/2., data.DT)

    pchanstr = r'' + ', '.join([PSTRINGS[c_name] for c_name in channel_names])
    ax = axes[0]
    # ax.set_title(pchanstr)
    ax.axison = False
    ax.text(0., 0., pchanstr,
             horizontalalignment='center', verticalalignment='center',
             rotation=45., fontsize=labelsize)
    ax.set_xlim((0.,1.))
    ax.set_ylim((-1.,1.))

    for ii in range(4):
        ax = noFrameAx(axes[ii+1])
        i0, i1 = i0s[ii], i1s[ii]
        # plot traces
        ax.plot(t_plot, v_dat.v_s[i0:i1], c='k', lw=lwidth, label=r'$Exp_{soma}$')
        ax.plot(t_plot, v_dat.v_d[i0:i1], c=pcolor, lw=lwidth, label=r'$Exp_{dend}$')
        ax.plot(t_plot, res['v_m'][0][i0:i1], c='k', lw=2*lwidth, alpha=.4, label=r'$Sim_{soma}$')
        ax.plot(t_plot, res['v_m'][1][i0:i1], c=pcolor, lw=2*lwidth, alpha=.4, label=r'$Sim_{soma}$')

        if zd:
            ax.set_ylim((-110., -50.))
        else:
            ax.set_ylim((-95., -50.))

        if ii == 0:
            if not zd:
                ax.set_yticks([-65., -70.])
                ax.set_yticklabels(['-65 mV', '-70 mV'])
            else:
                ax.set_yticks([-75., -80.])
                ax.set_yticklabels(['-75 mV', '-80 mV'])

        if ii ==3 and plot_legend:
            myLegend(ax)
            # draw x scalebar
            ypos = ax.get_ylim()[0]
            xpos = ax.get_xlim()[-1] - 40.
            sblen = 400.
            ax.plot([xpos-sblen,xpos], [ypos, ypos], 'k-', lw=2.*lwidth)
            ax.annotate(r'%.0f ms'%sblen,
                            xy=(xpos-sblen/2., ypos), xytext=(xpos-sblen/2., ypos - 4.),
                            size=labelsize, rotation=0, ha='center', va='center')
            # draw y scalebar
            ypos = ax.get_ylim()[0] + 5.
            xpos = ax.get_xlim()[-1]
            sblen = 10.
            ax.plot([xpos,xpos], [ypos, ypos+sblen], 'k-', lw=2.*lwidth)
            ax.annotate(r'%.0f mV'%sblen,
                            xy=(xpos, ypos+sblen/2.), xytext=(xpos+100., ypos+sblen/2.),
                            size=labelsize, rotation=90, ha='center', va='center')

    ax = myAx(axes[5])
    print('>>> parameters')
    for jj, pvals in enumerate(hall_of_fame):
        model_evaluator.setParameterValues(pvals)
        ps = model_evaluator.getParameterValuesAsDict()
        print(model_evaluator.toStrParameterValues())
        d2s = np.linspace(0., 400., int(1e3))
        for ii, c_name in enumerate(model_evaluator.channel_names):
            func = utils.expDistr(ps['d_'+c_name], ps['g0_'+c_name],
                                  g_max=model_evaluator.g_max[c_name])
            if jj == 0:
                ax.plot(d2s, func(d2s)*unit_conv, c=PCOLORS[c_name], lw=lwidth, label=PSTRINGS[c_name])
            else:
                ax.plot(d2s, func(d2s)*unit_conv, c=PCOLORS[c_name], lw=3.*lwidth, alpha=.07)
    if plot_legend:
        myLegend(ax)
    if plot_xlabel:
        ax.set_xlabel(r'Distance ($\mu$m)', fontsize=labelsize)
    else:
        ax.set_xticklabels([])
    if plot_ylabel:
        if units == 'pS/um2':
            ax.set_ylabel(r'Conductance (pS/$\mu$m$^2$)', fontsize=labelsize)
        else:
            ax.set_ylabel(r'Conductance ($\mu$S/cm$^2$)', fontsize=labelsize)
    if zd:
        ax.set_ylim((0.,2000.*unit_conv))
        ax.set_yticks([0.,1000.*unit_conv])
    else:
        ax.set_ylim((0.,6000.*unit_conv))
        ax.set_yticks([0.,3000.*unit_conv])

    if pshow:
        pl.tight_layout()
        pl.show()


def calcError(model_evaluator, hall_of_fame, v_dat):
    v_errs = []
    for pvals in hall_of_fame:
        fitness = model_evaluator.evaluate(pvals)
        v_errs.append(np.mean(fitness))
    return v_errs


def plotErrors(channel_names_list, zd_list, ax=None, qs=[.01,.5,.99], suffix='_predef'):
    global PSTRINGS

    full_tree, red_tree, full_locs, red_locs = data.reduceMorphology()
    sim_tree = red_tree.__copy__(new_tree=NeuronSimTree())

    v_qmins, v_meds, v_qmaxs = [], [], []
    ticklabels = []
    for channel_names, zd in zip(channel_names_list, zd_list):

        # load hall of fame
        file_name = utils.getFileName(channel_names, zd, suffix=suffix)
        file = open(file_name, 'rb')
        hall_of_fame = pickle.load(file)
        file.close()

        # measured data
        v_dat = data.DataContainer(with_zd=zd)

        # model
        model_evaluator = optimizer.ModelEvaluator(sim_tree, v_dat,
                                                    red_locs[0], red_locs[1],
                                                    channel_names=channel_names)

        # compute the errors
        v_errs = calcError(model_evaluator, hall_of_fame, v_dat)

        # compute quantiles
        v_qmin, v_med, v_qmax = np.quantile(v_errs, qs)
        v_qmins.append(v_qmin); v_meds.append(v_med); v_qmaxs.append(v_qmax)

        # tick label for this configuration
        tl = r'' + ', '.join([PSTRINGS[c_name] for c_name in channel_names])
        ticklabels.append(tl)

    v_meds = np.array(v_meds)
    v_errs = np.abs(np.array([v_qmins, v_qmaxs]) - v_meds[None,:])
    inds_h = np.where(np.logical_not(np.array(zd_list)))[0]
    inds_zd = np.where(np.array(zd_list))[0]
    n_h = len(inds_h)
    n_zd = len(inds_zd)
    ticklabels = [ticklabels[ii] for ii in inds_h]


    print(v_errs)

    if ax is None:
        pl.figure('v errors')
        ax = pl.gca()
        pshow = True
    else:
        pshow = False

    if n_h > 0:
        ax.bar(np.arange(n_h), v_meds[inds_h], width=.3, align='edge', yerr=v_errs[:,inds_h], color='Blue')
    if n_zd > 0:
        ax.bar(np.arange(n_zd), v_meds[inds_zd], width=-.3, align='edge', yerr=v_errs[:,inds_zd], color='Orange')

    ax.set_xticks(np.arange(np.max([n_zd, n_h])))
    ax.set_xticklabels(ticklabels, rotation=60., fontsize=labelsize)
    ax.set_ylabel(r'$V_{error}$ (mV)')


    if pshow:
        pl.show()


if __name__ == "__main__":
    ## plot trace example without zd ###########################################
    plotTrace(['L', 'K_ir', 'K_m35', 'h_u'], False)
    ############################################################################

    ## plot trace example with zd ##############################################
    # plotTrace(['L', 'K_ir', 'K_m35'], True)
    ############################################################################

    ## plot errors of each configuration #######################################
    # channel_names_list = [ ['L_c'],
    #                        ['L_c'],
    #                        ['L'],
    #                        ['L'],
    #                        ['L', 'K_ir'],
    #                        ['L', 'K_ir'],
    #                        ['L', 'K_m35'],
    #                        ['L', 'K_m35'],
    #                        ['L', 'K_ir', 'K_m35'],
    #                        ['L', 'K_ir', 'K_m35'],
    #                        ['L', 'h_u'],
    #                        ['L', 'K_ir', 'h_u'],
    #                        ['L', 'K_m35', 'h_u'],
    #                        ['L', 'K_ir', 'K_m35', 'h_u'],
    #                      ]
    # zd_list = [True,
    #            False,
    #            True,
    #            False,
    #            True,
    #            False,
    #            True,
    #            False,
    #            True,
    #            False,
    #            False,
    #            False,
    #            False,
    #            False
    #           ]
    # plotErrors(channel_names_list, zd_list)
    ############################################################################


