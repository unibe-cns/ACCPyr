import numpy as np

from bluepyopt.ephys.efeatures import EFeature

import data


################################################################################
def getFileName(channel_names, zd, suffix=''):
    """
    Auxililiary function to construct file name of optimization results
    """
    file_name = 'params/hof_' + '_'.join(channel_names)
    if zd: file_name += '_with_zd'
    file_name += suffix + '.p'
    return file_name
################################################################################


## ion channel distribution functions ##########################################
class expDistr(object):
    def __init__(self, d_scale, g_0, g_max=40000.):
        self.d_scale = d_scale
        self.g_0 = g_0
        self.g_max = g_max

    def __call__(self, d2s):
        v0 = self.g_0 * np.exp(self.d_scale * d2s)
        return np.clip(v0, 0., self.g_max)

class flatDistr(object):
    def __init__(self, p0):
        self.p0 = p0

    def __call__(self, d2s):
        return self.p0

class linDistr(object):
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1

    def __call__(self, d2s):
        return self.p0 + self.p1 * d2s

class quadDistr(object):
    def __init__(self, p0, p1, p2, pmax=10000.):
        self.p0, self.p1, self.p2, self.pmax = p0, p1, p2, pmax

    def __call__(self, d2s):
        v0 = self.p0 + self.p1 * d2s + self.p2 * d2s**2
        return np.clip(v0, 0., self.g_max)
################################################################################


## optimisation features #######################################################
def RMSE(arr1, arr2):
    if not isinstance(arr1, np.ndarray): arr1 = np.array(arr1)
    if not isinstance(arr2, np.ndarray): arr2 = np.array(arr2)
    return np.sqrt(np.mean((arr1 - arr2)**2))


class VeqFeature(EFeature):
    def __init__(self, data):
        self.data = data
        self.extractVeq(self.data.v_s, self.data.v_d, store=True)

    def extractVeq(self, v_s, v_d, store=False):
        t0s = np.array([data.T0, data.T1, data.T2, data.T3]) - data.T0
        t1s = np.array([data.T0, data.T1, data.T2, data.T3]) - 5.
        i0s = (t0s / self.data.dt).astype(int)
        i1s = (t1s / self.data.dt).astype(int)

        v_eq_s = np.mean([v_s[i0:i1] for i0, i1 in zip(i0s, i1s)])
        v_eq_d = np.mean([v_d[i0:i1] for i0, i1 in zip(i0s, i1s)])

        if store:
            self.v_eq_s, self.v_eq_d = v_eq_s, v_eq_d
        return v_eq_s, v_eq_d

    def calculate_score(self, responses):
        v_eq_s, v_eq_d = self.extractVeq(responses[0], responses[1])
        rmse = RMSE([v_eq_s, v_eq_d], [self.v_eq_s, self.v_eq_d])
        return rmse

class VStepFeature(EFeature):
    def __init__(self, data):
        self.data = data
        self.extractVStep(self.data.v_s, self.data.v_d, store=True)

    def extractVStep(self, v_s, v_d, store=False):
        t0s = np.array([data.T0, data.T1, data.T2, data.T3]) + 500.
        t1s = np.array([data.T0, data.T1, data.T2, data.T3]) + 595.
        i0s = (t0s / self.data.dt).astype(int)
        i1s = (t1s / self.data.dt).astype(int)

        v_step_s = np.array([np.mean(v_s[i0:i1]) for i0, i1 in zip(i0s, i1s)])
        v_step_d = np.array([np.mean(v_d[i0:i1]) for i0, i1 in zip(i0s, i1s)])

        if store:
            # print 'v_step_s =', v_step_s
            # print 'v_step_d =', v_step_d
            self.v_step_s, self.v_step_d = v_step_s, v_step_d
        return v_step_s, v_step_d

    def calculate_score(self, responses):
        v_step_s, v_step_d = self.extractVStep(responses[0], responses[1])
        rmse = RMSE(np.concatenate((v_step_s, v_step_d)), \
                    np.concatenate((self.v_step_s, self.v_step_d)))
        return rmse

class V01StepFeature(EFeature):
    def __init__(self, data):
        self.data = data
        self.extractV01Step(self.data.v_s, self.data.v_d, store=True)

    def extractV01Step(self, v_s, v_d, store=False):
        t0s = np.array([data.T0, data.T1]) + 500.
        t1s = np.array([data.T0, data.T1]) + 595.
        i0s = (t0s / self.data.dt).astype(int)
        i1s = (t1s / self.data.dt).astype(int)

        v_step_s = np.array([np.mean(v_s[i0:i1]) for i0, i1 in zip(i0s, i1s)])
        v_step_d = np.array([np.mean(v_d[i0:i1]) for i0, i1 in zip(i0s, i1s)])

        if store:
            # print 'v_step_s =', v_step_s
            # print 'v_step_d =', v_step_d
            self.v_step_s, self.v_step_d = v_step_s, v_step_d
        return v_step_s, v_step_d

    def calculate_score(self, responses):
        v_step_s, v_step_d = self.extractV01Step(responses[0], responses[1])
        rmse = RMSE(np.concatenate((v_step_s, v_step_d)), \
                    np.concatenate((self.v_step_s, self.v_step_d)))
        return rmse


class V23StepFeature(EFeature):
    def __init__(self, data):
        self.data = data
        self.extractV23Step(self.data.v_s, self.data.v_d, store=True)

    def extractV23Step(self, v_s, v_d, store=False):
        t0s = np.array([data.T2, data.T3]) + 500.
        t1s = np.array([data.T2, data.T3]) + 595.
        i0s = (t0s / self.data.dt).astype(int)
        i1s = (t1s / self.data.dt).astype(int)

        v_step_s = np.array([np.mean(v_s[i0:i1]) for i0, i1 in zip(i0s, i1s)])
        v_step_d = np.array([np.mean(v_d[i0:i1]) for i0, i1 in zip(i0s, i1s)])

        if store:
            self.v_step_s, self.v_step_d = v_step_s, v_step_d
        return v_step_s, v_step_d

    def calculate_score(self, responses):
        v_step_s, v_step_d = self.extractV23Step(responses[0], responses[1])
        rmse = RMSE(np.concatenate((v_step_s, v_step_d)), \
                    np.concatenate((self.v_step_s, self.v_step_d)))
        return rmse


class TraceFeature(EFeature):
    def __init__(self, data):
        self.data = data

    def calculate_score(self, responses):
        v_s, v_d = responses[0], responses[1]
        rmse = RMSE(np.concatenate((v_s, v_d)), \
                    np.concatenate((self.data.v_s, self.data.v_d)))
        return rmse

class DecayFeature(EFeature):
    def __init__(self, data, t_margin=10.):
        self.data = data
        self.t_margin = t_margin
        self.extractVDecay(self.data.v_s, self.data.v_d, store=True)

    def extractVDecay(self, v_s, v_d, store=False):
        t0s = np.array([data.T0, data.T1, data.T2, data.T3]) + data.DUR + self.t_margin
        t1s = np.array([data.T0, data.T1, data.T2, data.T3]) + data.DUR + 270.
        i0s = (t0s / self.data.dt).astype(int)
        i1s = (t1s / self.data.dt).astype(int)

        v_decay_s = np.array([v_s[i0:i1] - v_s[i1] for i0, i1 in zip(i0s, i1s)])
        v_decay_d = np.array([v_d[i0:i1] - v_d[i1] for i0, i1 in zip(i0s, i1s)])

        if store:
            self.v_decay_s, self.v_decay_d = v_decay_s, v_decay_d
        return v_decay_s, v_decay_d

    def calculate_score(self, responses):
        v_decay_s, v_decay_d = self.extractVDecay(responses[0], responses[1])
        rmse = RMSE(np.concatenate((v_decay_s, v_decay_d)), \
                    np.concatenate((self.v_decay_s, self.v_decay_d)))
        return rmse

class AttFeature(EFeature):
    def __init__(self, data):
        self.data = data
        self.v_eq_f = VeqFeature(data)
        self.v_step_f = VStepFeature(data)
        self.att_d2s, self.att_s2d = self.calcAttenuation(data.v_s, data.v_d, store=True)

    def calcAttenuation(self, v_s, v_d, store=False):
        v_step_s, v_step_d = self.v_step_f.extractVStep(v_s, v_d)
        v_eq_s, v_eq_d = self.v_eq_f.extractVeq(v_s, v_d)
        v_diff_s = v_step_s - v_eq_s
        v_diff_d = v_step_d - v_eq_d
        att_d2s = v_diff_s[:2] / v_diff_d[:2]
        att_s2d = v_diff_d[2:] / v_diff_s[2:]

        if store:
            self.att_d2s, self.att_s2d = att_d2s, att_s2d

        return att_d2s, att_s2d

    def str(self, v_s, v_d):
        att_d2s, att_s2d = self.calcAttenuation(v_s, v_d)

        rstr =  '--- attenuation sim (exp) ---\n' + \
                '--> I_in negative\n' + \
                's2d = %.2f (%.2f)'%(att_s2d[0], self.att_s2d[0]) + '\n' + \
                'd2s = %.2f (%.2f)'%(att_d2s[0], self.att_d2s[0]) + '\n' + \
                '--> I_in positive\n' + \
                's2d = %.2f (%.2f)'%(att_s2d[1], self.att_s2d[1]) + '\n' + \
                'd2s = %.2f (%.2f)'%(att_d2s[1], self.att_d2s[1]) + '\n' + \
                '-----------------------------\n'
        return rstr

    def calculate_score(self, responses):
        v_s, v_d = responses[0], responses[1]
        att_d2s, att_s2d = self.calcAttenuation(v_s, v_d)
        rmse = RMSE(att_d2s, self.att_d2s) + RMSE(att_s2d, self.att_s2d)
        return 10.*rmse


class AttFeature_(AttFeature):
    def __init__(self, att_d2s, att_s2d, data):
        self.v_eq_f = VeqFeature(data)
        self.v_step_f = VStepFeature(data)
        self.att_d2s, self.att_s2d = att_d2s, att_s2d
################################################################################