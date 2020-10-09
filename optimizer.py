import numpy as np

from bluepyopt.deapext.optimisations import IBEADEAPOptimisation
from bluepyopt.ephys.objectives import WeightedSumObjective, SingletonObjective
from bluepyopt.parameters import Parameter
from bluepyopt.evaluators import Evaluator

from neat import NeuronSimTree

import pickle

from channels import channelcollection
import utils, data


## optimization ################################################################
# trial run parameter values
MAX_ITER = 10
N_OFFSPRING = 2
# full optimization parameter values
# MAX_ITER = 100
# N_OFFSPRING = 100
################################################################################


## model evaluator for optimization ############################################
class ModelEvaluator(Evaluator):
    def __init__(self, sim_tree, v_dat,
                       loc_soma, loc_dend,
                       channel_names=['L'],
                       mode='fit'):
        """
        if mode is fit, check bounds, if mode is evaluate, dont check bounds
        """
        self.sim_tree = sim_tree
        self.channel_names = channel_names
        # injection sites
        self.loc_soma, self.loc_dend = loc_soma, loc_dend
        # params
        self.dt, self.dur, self.t_max = data.DT, data.DUR, data.T_MAX
        self.t0, self.t1, self.t2, self.t3 = data.T0, data.T1, data.T2, data.T3
        self.a0, self.a1, self.a2, self.a3 = data.A0, data.A1, data.A2, data.A3
        self.channel_names = channel_names

        # constant params
        self.r_a = 113./1e6
        self.g_max = {'L': 20.*1e2, 'L_c': 20.*1e2,
                      'K_ir': 40.*1e2, 'K_m': 40.*1e2, 'K_m35': 40.*1e2,
                      'h_u': 60.*1e2, 'h_HAY': 60.*1e2,
                      'Na_p': 100.*1e2, 'NaP': 100.*1e2}
        # define fit parameters
        self._defineFitObjects(v_dat)

        # don't check bounds
        if mode == 'evaluate':
            for p in self.params:
                p.bounds = None

    def _defineFitObjects(self, v_dat):
        # fitness evaluator
        features = [utils.VeqFeature(v_dat), utils.VStepFeature(v_dat),  utils.TraceFeature(v_dat)]
        self.objectives = [SingletonObjective('V_eq', features[0]),
                           SingletonObjective('V_step', features[1]),
                           SingletonObjective('V_trace', features[2])]
        # parameters
        self.params = [Parameter('d_c_m', value=0.001, bounds=[0., 0.01]),
                       Parameter('c_m_0', value=1., bounds=[0.50, 1.50])]
        for c_name in self.channel_names:
            if c_name == 'K_m' or c_name == 'K_m35' or c_name == 'K_ir':
                params = [Parameter('d_'+c_name, value=-1./200., bounds=[-0.1, 0.]),
                          Parameter('g0_'+c_name, value=40.*1e2, bounds=[0., 10000.]),
                          Parameter('e_r_'+c_name, value=-85., bounds=[-95.,-80.])]
            elif c_name == 'h_HAY' or c_name == 'h_u':
                params = [Parameter('d_'+c_name, value=1./200., bounds=[0.0, 0.1]),
                          Parameter('g0_'+c_name, value=0.0099*1e2, bounds=[0., 10000.]),
                          Parameter('e_r_'+c_name, value=-40., bounds=[-50.,-30.])]
            elif c_name == 'Na_p' or c_name == 'NaP':
                params = [Parameter('d_'+c_name, value=0., bounds=[-0.0001, 0.0001]),
                          Parameter('g0_'+c_name, value=0.01*1e2, bounds=[0.,10000.]),
                          Parameter('e_r_'+c_name, value=50., bounds=[40.,60.])]
            # elif c_name == 'L':
            #     params = [Parameter('d_L', value=1./200., bounds=[0.0, 0.05]),
            #               Parameter('g0_L', value=0.40*1e2, bounds=[0., 300.]),
            #               Parameter('e_0_L', value=-90., bounds=[-100., -50.]),
            #               Parameter('e_c_L', value=0., bounds=[-1./15, 1./15])]
            elif c_name == 'L':
                params = [Parameter('d_L', value=1./1000., bounds=[0.0, 0.05]),
                          Parameter('g0_L', value=0.40*1e2, bounds=[0., 300.]),
                          Parameter('e_0_L', value=-90., bounds=[-100., -50.])]
            elif c_name == 'L_c':
                params = [Parameter('d_L_c', value=1./1000., bounds=[0.0, 0.05]),
                          Parameter('g0_L_c', value=0.40*1e2, bounds=[0., 300.]),
                          Parameter('e_0_L_c', value=-90., bounds=[-100., -50.]),
                          Parameter('e_c_L_c', value=0., bounds=[-1./15, 1./15])]

        # # original params
        # self.params = [Parameter('d_c_m', value=0.001, bounds=[0., 0.01]),
        #                Parameter('c_m_0', value=1., bounds=[0.95, 1.05])]
        # for c_name in self.channel_names:
        #     if c_name == 'K_m' or c_name == 'K_m35' or c_name == 'K_ir':
        #         params = [Parameter('d_'+c_name, value=-1./100., bounds=[-0.1, 0.]),
        #                   Parameter('g0_'+c_name, value=40.*1e2, bounds=[0., 10000.]),
        #                   Parameter('e_r_'+c_name, value=-85., bounds=[-95.,-80.])]
        #     elif c_name == 'h_HAY' or c_name == 'h_u':
        #         params = [Parameter('d_'+c_name, value=1./100., bounds=[0.0, 0.1]),
        #                   Parameter('g0_'+c_name, value=0.0099*1e2, bounds=[0., 10000.]),
        #                   Parameter('e_r_'+c_name, value=-40., bounds=[-50.,-30.])]
        #     elif c_name == 'Na_p' or c_name == 'NaP':
        #         params = [Parameter('d_'+c_name, value=0., bounds=[-0.0001, 0.0001]),
        #                   Parameter('g0_'+c_name, value=0.01*1e2, bounds=[0.,10000.]),
        #                   Parameter('e_r_'+c_name, value=50., bounds=[40.,60.])]
        #     # elif c_name == 'L':
        #     #     params = [Parameter('d_L', value=1./200., bounds=[0.0, 0.05]),
        #     #               Parameter('g0_L', value=0.40*1e2, bounds=[0., 300.]),
        #     #               Parameter('e_0_L', value=-90., bounds=[-100., -50.]),
        #     #               Parameter('e_c_L', value=0., bounds=[-1./15, 1./15])]
        #     elif c_name == 'L':
        #         params = [Parameter('d_L', value=1./200., bounds=[0.0, 0.05]),
        #                   Parameter('g0_L', value=0.40*1e2, bounds=[0., 300.]),
        #                   Parameter('e_0_L', value=-90., bounds=[-100., -50.])]
        #     elif c_name == 'L_c':
        #         params = [Parameter('d_L_c', value=1./200., bounds=[0.0, 0.05]),
        #                   Parameter('g0_L_c', value=0.40*1e2, bounds=[0., 300.]),
        #                   Parameter('e_0_L_c', value=-90., bounds=[-100., -50.]),
        #                   Parameter('e_c_L_c', value=0., bounds=[-1./15, 1./15])]


            else:
                warnings.warn('unrecognized ion channel \'' + c_name +'\'( choose from ' + \
                              ' '.join(['L', 'K_m', 'K_m35', 'K_ir', 'h_HAY', 'h_u']), + \
                              '), ignoring current channel.')
            self.params.extend(params)

    def evalFitness(self, responses):
        return [obj.calculate_score(responses) for obj in self.objectives]

    def getParameterValues(self):
        return [p.value for p in self.params]

    def setParameterValues(self, values):
        if values is None:
            values = self.getParameterValues()
        if isinstance(values, list):
            for p, v in zip(self.params, values): p.value = v
        elif isinstance(values, dict):
            for p in self.params: p.value = values[p.name]
        else:
            raise TypeError('``values`` must be `list` or `dict`')

    def getParameterValuesAsDict(self):
        return {p.name: p.value for p in self.params}

    def toStrParameterValues(self, values=None):
        rstr = 'Parametervalues =\n'
        if values is None:
            values = [p.value for p in self.params]
        if isinstance(values, list):
            for p, v in zip(self.params, values):
                rstr += '    > ' + p.name + ' = %.5f\n'%v
        elif isinstance(values, dict):
            for p in self.params:
                rstr += '    > ' + p.name + ' = %.5f\n'%values[p.name]
        return rstr

    def toStrFitness(self, responses):
        fitness = self.evalFitness(responses)
        rstr = 'Fitness =\n'
        for ii, ff in enumerate(fitness):
            rstr += '    > f_%d = %.5f\n'%(ii,ff)
        return rstr

    def getTreeWithParams(self, new_tree=None):
        ps = self.getParameterValuesAsDict()
        # set the physiology parameters of this tree
        sim_tree = self.sim_tree.__copy__(new_tree=new_tree)
        sim_tree.treetype = 'original'
        # capacitance
        c_m_distr = utils.linDistr(ps['c_m_0'], ps['d_c_m'])
        sim_tree.setPhysiology(c_m_distr, self.r_a)
        # membrane current parameters
        for ii, c_name in enumerate(self.channel_names):
            g_func = utils.expDistr(ps['d_'+c_name], ps['g0_'+c_name], g_max=self.g_max[c_name])
            if c_name != 'L' and c_name != 'L_c':
                e_r = ps['e_r_'+c_name]
                # add the current
                chan = eval('channelcollection.' + c_name + '()')
                sim_tree.addCurrent(chan, g_func, e_r)
            else:
                if c_name == 'L_c':
                    e_func = utils.linDistr(ps['e_0_L_c'], ps['e_c_L_c'])
                else:
                    e_func = lambda x: ps['e_0_L']
                # add the current
                for node in sim_tree:
                    d2s = sim_tree.pathLength({'node': node.index, 'x': .5}, (1., 0.5))
                    g_l = g_func(d2s)
                    e_l = e_func(d2s)
                    node._addCurrent('L', g_l, e_l)

        return sim_tree

    def runSim(self):
        '''
        Format for args:
            [c_m, r_a] +
            [d_scale, g_0, g_max] for expDistr for each conductance channel in self.channel_names +
            [e_0, e_1] for the leak potential
        '''
        sim_tree = self.getTreeWithParams()
        # initialize the simulation
        sim_tree.setCompTree()
        sim_tree.treetype = 'computational'
        sim_tree.initModel(dt=self.dt, t_calibrate=200.)
        # add Iclamps
        sim_tree.addIClamp(self.loc_dend, self.a0, self.t0, self.dur)
        sim_tree.addIClamp(self.loc_dend, self.a1, self.t1, self.dur)
        sim_tree.addIClamp(self.loc_soma, self.a2, self.t2, self.dur)
        sim_tree.addIClamp(self.loc_soma, self.a3, self.t3, self.dur)
        # set recorders
        sim_tree.storeLocs([self.loc_soma, self.loc_dend], name='rec locs')

        # run simulation
        res = sim_tree.run(self.t_max, pprint=False)

        sim_tree.deleteModel()

        return res

    def evaluate_with_lists(self, param_values=None):
        return self.evaluate(param_values)

    def evaluate(self, param_values, pprint=True):
        self.setParameterValues(values=param_values)
        res = self.runSim()
        fitness = self.evalFitness(res['v_m'][:,:-1])
        if pprint:
            print('>>> fitness =', fitness)
            # print '>>> ' + self.toStrParameterValues()
        return fitness


class AttenuationEvaluator(ModelEvaluator):
    def __init__(self, sim_tree, f_d2s, f_s2d,
                       loc_soma, loc_dend, mode='fit'):
        """
        Only optimizes h-current

        if mode is fit, check bounds, if mode is evaluate, dont check bounds
        """
        self.sim_tree = sim_tree
        # injection sites
        self.loc_soma, self.loc_dend = loc_soma, loc_dend
        # params
        self.dt, self.dur, self.t_max = data.DT, data.DUR, data.T_MAX
        self.t0, self.t1, self.t2, self.t3 = data.T0, data.T1, data.T2, data.T3
        self.a0, self.a1, self.a2, self.a3 = data.A0, data.A1, data.A2, data.A3

        self.mode = mode

        # define fit parameters
        self._defineFitObjects(f_d2s, f_s2d)

        # don't check bounds
        if mode == 'evaluate':
            for p in self.params:
                p.bounds = None

    def _defineFitObjects(self, f_d2s, f_s2d):
        # reference attenuation
        v_dat = data.DataContainer(with_zd=True)
        att_f = utils.AttFeature(v_dat)
        att_ref_d2s = att_f.att_d2s * f_d2s
        att_ref_s2d = att_f.att_s2d * f_d2s
        # fitness evaluator
        features = [utils.AttFeature_(att_ref_d2s, att_ref_s2d, v_dat)]
        self.objectives = [SingletonObjective('Att', features[0])]
        # parameters
        self.params = [
                       Parameter('g_h_0', value=0., bounds=[0.,50000.]),
                       Parameter('g_h_1', value=200., bounds=[0.,50000.]),
                       Parameter('g_h_2', value=2000., bounds=[0.,50000.]),
                       Parameter('g_h_3', value=5000., bounds=[0.,50000.]),
                       Parameter('e_r_h', value=-40., bounds=[-50.,-30.]),
                       Parameter('g_h_b', value=0., bounds=[0.,50000.]),
                       ]

    def _h_distr_func(self, x, ds=[0., 250., 500., 750.]):
        ps = self.getParameterValues()

        if x <= ds[1]:
            d0 = ds[0]; d1 = ds[1]
            p0 = ps[0]; p1 = ps[1]
        elif x > ds[1] and x < ds[2]:
            d0 = ds[1]; d1 = ds[2]
            p0 = ps[1]; p1 = ps[2]
        else:
            d0 = ds[2]; d1 = ds[3]
            p0 = ps[2]; p1 = ps[3]

        return p0 + (p1 - p0) / (d1 - d0) * (x - d0)

    def getTreeWithParams(self, new_tree=None):
        ps = self.getParameterValuesAsDict()
        # set the physiology parameters of this tree
        sim_tree = self.sim_tree.__copy__(new_tree=new_tree)
        sim_tree.treetype = 'original'
        # h-current distribution
        h_u = channelcollection.h_u()
        sim_tree.addCurrent(h_u, self._h_distr_func, ps['e_r_h'], node_arg='apical')

        sim_tree.addCurrent(h_u, ps['g_h_b'], ps['e_r_h'], node_arg='basal')
        sim_tree.addCurrent(h_u, ps['g_h_b'], ps['e_r_h'], node_arg=[sim_tree[1]])

        return sim_tree
################################################################################


def optimize(evaluator):
    global MAX_ITER, N_OFFSPRING

    optimisation = IBEADEAPOptimisation(evaluator=evaluator,
                                        offspring_size=N_OFFSPRING, map_function=map)
    final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=MAX_ITER)

    return final_pop, hall_of_fame, logs, hist


def optimizeModel(channel_names=None, zd=False, suffix=''):
    """
    Optimizes the morphology equipped with channels in `channel_names` to
    recordings with or without ZD

    Parameters
    ----------
    channel_names: list of str
        Choose channel names from from {'L', 'K_m', 'K_m35', 'K_ir', 'h_HAY', 'h_u'}
    zd: bool
        True for data with ZD, false for data without ZD
    """
    global MAX_ITER, N_OFFSPRING

    if channel_names is None:
        channel_names = ['L', 'K_ir', 'K_m35', 'h_u']
    file_name = utils.getFileName(channel_names, zd, suffix=suffix)

    full_tree, red_tree, full_locs, red_locs = data.reduceMorphology()
    sim_tree = red_tree.__copy__(new_tree=NeuronSimTree())

    # measured data
    v_dat = data.DataContainer(with_zd=zd)
    model_evaluator = ModelEvaluator(sim_tree, v_dat, red_locs[0], red_locs[1],
                                     channel_names=channel_names)

    final_pop, hall_of_fame, logs, hist = optimize(model_evaluator)

    # save hall of fame
    file = open(file_name, 'wb')
    pickle.dump(hall_of_fame, file)
    file.close()


if __name__ == "__main__":
    optimizeModel(channel_names=['L'], zd=False, suffix='_test')