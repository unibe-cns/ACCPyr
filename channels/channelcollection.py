import sympy as sp

from neat import IonChannel


class K_ir(IonChannel):
    '''
    L2/3 Pyr (Branco et al., 2010)
    '''
    def define(self):
        self.ion = 'k'
        self.p_open = 'm'
        # asomptotic state variable functions
        self.varinf = {'m': '1. / (1. + exp((v + 82.) / 13.))'}
        self.tauinf = {'m': '6.*2.'}


class K_m(IonChannel):
    '''
    L2/3 Pyr (Branco et al., 2010)
    '''
    def define(self):
        self.ion = 'k'
        self.p_open = 'n'
        # activation functions
        self.alpha = {'n': ' 0.001 * (v + 30.) / (1. - exp(-(v + 30.) / 9.))'}
        self.beta  = {'n': '-0.001 * (v + 30.) / (1. - exp( (v + 30.) / 9.))'}
        self.q10 = 3.21


class K_m35(IonChannel):
    '''
    m-type potassium channel
    Used in (Mengual et al., 2010)
    '''
    def define(self):
        self.ion = 'k'
        self.p_open = 'n'
        # activation functions
        self.alpha = {'n': ' 0.001 * (v + 30.) / (1. - exp(-(v + 30.) / 9.))'}
        self.beta  = {'n': '-0.001 * (v + 30.) / (1. - exp( (v + 30.) / 9.))'}
        self.q10 = 2.71


class h_u(IonChannel):
    '''
    hcn channel
    Used in (Mengual et al., 2019)
    '''
    def define(self):
        self.p_open = 'q'
        # activation functions
        self.alpha = {'q': '0.001*6.43* (v+154.9) / (exp((v+154.9) / 11.9) - 1.)'}
        self.beta  = {'q': '0.001*193 * exp(v/33.1)'}


class h_HAY(IonChannel):
    '''
    Hcn channel from (Kole, Hallermann and Stuart, 2006)
    Used in (Hay, 2011)
    '''
    def define(self):
        self.p_open = 'm'
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha['m'] = '0.001 * 6.43 * (v + 154.9) / (exp((v + 154.9) / 11.9) - 1.)'
        self.beta['m']  = '0.001 * 193. * exp(v / 33.1)'


class Na_p(IonChannel):
    '''
    Derived by (Hay, 2011) from (Magistretti and Alonso, 1999)
    Used in (Hay, 2011)
    '''
    def define(self):
        self.ion = 'na'
        # channel open probability
        self.p_open = 'm**3 * h'
        # activation functions
        self.varinf = {'m': '1. / (1. + exp(-(v + 52.6) / 4.6))',
                       'h': '1. / (1. + exp( (v + 48.8) / 10.))'}
        # non-standard time-scale definition
        v = sp.symbols('v')
        alpha, beta = {}, {}
        alpha['m'] =   0.182   * (v + 38. ) / (1. - sp.exp(-(v + 38. ) / 6.  , evaluate=False)) #1/ms
        beta['m']  = - 0.124   * (v + 38. ) / (1. - sp.exp( (v + 38. ) / 6.  , evaluate=False)) #1/ms
        alpha['h'] = - 2.88e-6 * (v + 17. ) / (1. - sp.exp( (v + 17. ) / 4.63, evaluate=False)) #1/ms
        beta['h']  =   6.94e-6 * (v + 64.4) / (1. - sp.exp(-(v + 64.4) / 2.63, evaluate=False)) #1/ms
        self.tauinf = {'m': 6./(alpha['m'] + beta['m']),
                       'h': 1./(alpha['h'] + beta['h'])}
        # temperature factor
        self.q10 = 2.95


class NaP(IonChannel):
    '''
    Purkinje Cell (Miyasho et al., 2001)
    '''
    def define(self):
        self.ion = 'na'
        self.p_open = 'm**3'
        # activation functions
        self.alpha, self.beta = {}, {}
        self.alpha['m'] = '200. / (1. + exp((v-18.) / (-16.)))'
        self.beta['m']  = '25.  / (1. + exp((v+58.) / 8.    ))'
