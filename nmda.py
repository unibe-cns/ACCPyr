import numpy as np
from neuron import h

import neat
from neat import NeuronSimTree, MorphLoc

import copy


# simulation
T_MAX = 300.
DT = 0.1
# EXPERIMENT
DELTA_T = 20. # ms
T0 = 60. # ms
# physiology
EL = -75.           # mV

# AMPA synapse parameters
TAU_AMPA = 2.       # ms
# NMDA synapse parameters
E_REV_NMDA = 5.     # mV
C_MG = 1.           # mM
DUR_REL = 0.5       # ms
AMP_REL = 2.        # mM


def calcAmpWidthSurface(v_arr, t0, dt=DT, teps=3.):
    """
    compute amplitude, width and surface of NMDA spike from voltage trace
    """
    i0 = int(t0/dt)
    i_ = int(teps/dt)
    v_eq = v_arr[0]

    amp   = np.max(v_arr[i0+i_:]) - v_eq
    width = dt * np.where(v_arr[i0:] > (amp/2.+ v_eq))[0][-1]
    surf  = dt * (np.sum(v_arr[i0:] - v_eq))

    return amp, width, surf


def subtractAP(v_m, v_ap, ix_spike, ix_loc):
    """
    Subtracts AP waveform in `v_ap` from AP waveform in `v_m`. Waveforms are
    aligned at the peak.

    `ix_spike` gives first and last index of window in which spike occured
    `ix_loc` gives location where peak time is measured
    """

    i_peak_m  = ix_spike[0] + np.argmax(v_m[ix_loc, ix_spike[0]:ix_spike[1]])
    i_peak_ap = np.argmax(v_ap[ix_loc])

    i0 = i_peak_m - i_peak_ap
    i1 = i0 + len(v_ap[ix_loc])
    i2 = min(len(v_m[ix_loc])-i0, len(v_ap[ix_loc]))

    v_m_ = copy.deepcopy(v_m)
    v_m_[:,i0:i1] -= v_ap[:,:i2]

    return v_m_


def deviationThrFromLinear(arr, f_eps=1.2):
    """
    Compute the threshold by finding where it has deviated by more than a factor
    `f_eps` from linear.
    """
    assert len(arr) > 2

    x_arr = np.arange(len(arr))

    import sklearn.linear_model as lm
    lr = lm.LinearRegression()
    lr.fit(x_arr[:,None], arr, np.exp(-x_arr))
    f_arr = lr.coef_[0] * x_arr + lr.intercept_

    try:
        ix_max = np.where((arr[1:]-f_arr[0]) > f_eps * (f_arr[1:]-f_arr[0]))[0][0]
        ix_max += 1
    except IndexError:
        ix_max = len(arr)-1

    return ix_max


class NMDASimTree(NeuronSimTree):
    def __init__(self, **kwargs):
        super(NMDASimTree, self).__init__(**kwargs)

        self.pres = []
        self.nmdas = []

    def deleteModel(self):
        super(NMDASimTree, self).deleteModel()
        self.pres = []
        self.nmdas = []

    def addAMPASynapse(self, loc, g_max, tau):
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.AlphaSynapse(self.sections[loc['node']](loc['x']))
        syn.tau = tau
        syn.gmax = g_max
        # store the synapse
        self.syns.append(syn)

        return len(self.syns)-1

    def addNMDASynapse(self, loc, g_max, e_rev, c_mg, dur_rel, amp_rel):
        loc = MorphLoc(loc, self)
        # create the synapse
        syn = h.NMDA_Mg_T(self.sections[loc['node']](loc['x']))
        syn.gmax = g_max
        syn.Erev = e_rev
        syn.mg = c_mg
        # create the presynaptic segment for release
        pre = h.Section(name='pre %d'%len(self.pres))
        pre.insert('release_BMK')
        pre(0.5).release_BMK.dur = dur_rel
        pre(0.5).release_BMK.amp = amp_rel
        # connect
        h.setpointer(pre(0.5).release_BMK._ref_T, 'C', syn)
        # store the synapse
        self.nmdas.append(syn)
        self.pres.append(pre)

        return len(self.nmdas) - 1

    def setSpikeTime(self, syn_index_ampa, syn_index_nmda, spike_time):
        spk_tm = spike_time + self.t_calibrate
        # add spike for AMPA synapse
        self.syns[syn_index_ampa].onset = spk_tm
        # add spike for NMDA synapse
        self.pres[syn_index_nmda](0.5).release_BMK.delay = spk_tm

    def addCombinedSynapse(self, loc, g_max_ampa, g_max_nmda):
        global TAU_AMPA, E_REV_NMDA, C_MG, DUR_REL, AMP_REL
        # ampa synapse
        syn_idx_ampa = self.addAMPASynapse(loc, g_max_ampa, TAU_AMPA)
        # nmda synapse
        syn_idx_nmda = self.addNMDASynapse(loc, g_max_nmda, E_REV_NMDA, C_MG, DUR_REL, AMP_REL)

        return syn_idx_ampa, syn_idx_nmda

    def setActivation(self, loc, n_syn, g_max_ampa, g_max_nmda, t0=T0, delta_t=DELTA_T):
        for ii in range(n_syn):
            syn_idx_ampa, syn_idx_nmda = self.addCombinedSynapse(loc, g_max_ampa, g_max_nmda)
            self.setSpikeTime(syn_idx_ampa, syn_idx_nmda, t0)
        for ii in range(n_syn):
            syn_idx_ampa, syn_idx_nmda = self.addCombinedSynapse(loc, g_max_ampa, g_max_nmda)
            self.setSpikeTime(syn_idx_ampa, syn_idx_nmda, t0+delta_t)

    @neat.trees.morphtree.computationalTreetypeDecorator
    def runExperiment(self, loc, g_max_ampa, g_max_nmda, n_syn=10, with_ap=False, delta_t=0., loc_=None, n_syn_=20):
        """
        Simulate the experiment with `n_syn` synapses at `loc`

        `with_ap` as ``True`` elicits ap with strong current pulse at soma
        """
        global EL, T_MAX, DT, T0, DELTA_T
        loc = MorphLoc(loc, self)

        self.initModel(dt=DT, t_calibrate=200., v_init=EL, factor_lambda=1.)
        # add the synapses
        self.setActivation(loc, n_syn, g_max_ampa, g_max_nmda, t0=T0)
        if loc_ is not None:
            loc_ = MorphLoc(loc_, self)
            self.setActivation(loc_, n_syn_, g_max_ampa, g_max_nmda, t0=T0-DELTA_T)
        # add current clap
        if with_ap:
            self.addIClamp((1,.5), 4., T0+delta_t, 1.)
        # set recording locs
        rec_locs = [(1, .5), loc]
        if loc_ is not None:
            rec_locs.append(loc_)
        self.storeLocs(rec_locs, name='rec locs')
        # run the simulation
        res = self.run(T_MAX, pprint=True)
        # delete the model
        self.deleteModel()

        return res

    def extractAP(self, loc, delta_t=0):
        """
        Extract action potential waveform
        """
        res_ap = self.runExperiment(loc, n_syn=0, with_ap=True, delta_t=delta_t)
        v_ap = res_ap['v_m'] - res_ap['v_m'][:,0:1]

        i0 = int(T0/DT)
        i1 = int(200/DT) + i0

        return v_ap[:,i0:i1]

    def findNMDAThreshold(self, loc, n_syns, g_max_ampa, g_max_nmda, delta_t=0.,
                          with_TTX=False, with_ap=False, at_soma=False, pplot=False,
                          loc_=None, n_syn_=20):
        """
        Extrac nmda spike threshold

        Returns
        -------
        n_syn_thr: int
            threshold number of synapses to activate
        res_nmda: dict of np.ndarray
            contains 'amp', 'width' and 'surf' of waveform elicited by second
            stimulus for each activation level
        res_sim: list of dict
            contains the voltage traces for each simulation
        at_soma: bool
            If ``True``, the threshold is measured at the soma. Otherwise at
            the dendritic synapse.
        """
        global T0, DELTA_T, DT
        assert 0 not in n_syns

        lll = MorphLoc(loc, self)
        print("\n--> Distance to soma = %.2f um \n"%self.distancesToSoma([lll])[0])

        if with_TTX:
            self.addTTX()

        ix_thr = 0 if at_soma else 1
        # extract baseline AP
        if with_ap:
            v_ap = self.extractAP(loc)
            ix_spike = (int((T0+delta_t)/DT), int((T0+delta_t+5.)/DT))

        res_sim = []
        res_nmda = {'amp': [],
                    'width': [],
                    'surf': []}
        for nn in n_syns:
            res = self.runExperiment(loc, g_max_ampa, g_max_nmda, n_syn=nn, delta_t=delta_t,
                                     with_ap=with_ap, loc_=loc_, n_syn_=n_syn_)
            if with_ap:
                res['v_m_'] = subtractAP(res['v_m'], v_ap, ix_spike, ix_thr)
            else:
                res['v_m_'] = copy.deepcopy(res['v_m'])
            res_sim.append(res)

            amp, width, surf = calcAmpWidthSurface(res['v_m_'][ix_thr], T0+DELTA_T, dt=DT)
            res_nmda['amp'].append(amp)
            res_nmda['width'].append(width)
            res_nmda['surf'].append(surf)

            if pplot:
                pl.figure()
                ax = pl.subplot(121)
                ax.set_title('soma')
                ax.plot(res['t'], res['v_m'][0], 'b')
                ax.plot(res['t'], res['v_m_'][0], 'r--')

                ax = pl.subplot(122)
                ax.set_title('dend')
                ax.plot(res['t'], res['v_m'][1], 'b')
                ax.plot(res['t'], res['v_m_'][1], 'r--')

                pl.show()

        for key, val in res_nmda.items(): res_nmda[key] = np.array(val)

        # nmda threshold as steepest surface increase
        n_syn_thr = deviationThrFromLinear(res_nmda['surf'])

        return n_syn_thr, res_nmda, res_sim