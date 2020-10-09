import numpy as np

from neat import MorphTree


## parameters of the recordings ################################################
# sampling interval of the recordings [ms]
DT = 0.125
# duration of the recordings [ms]
T_MAX = 4000.
# duration of the input current steps [ms]
DUR = 600.
# onset timeings of the input current steps [ms]
T0, T1, T2, T3 = 100., 1100., 2100., 3100.
# amplitudes of the recording steps
A0, A1, A2, A3 = -.3, .1, -.3, .1
# somatic resp. dendritic recording site
LOC_SOMA = (1, 0.5)
LOC_DEND = (188, 0.6)
################################################################################


## container for experiemental data ############################################
def reduceMorphology():
    """
    Reduces the number of nodes on the morphology for computational efficiency

    Returns
    -------
    m_tree: `neat.MorphTree`
        The original morphology
    morph_tree: `neat.MorphTree`
        The reduced morphology
    locs: list of locations
        The recording locations in the coordinates of the original morphology
    morph_locs: list of locations
        The recording locations in the coordinates of the reduced tree
    """
    m_tree = MorphTree(file_n='datafiles/ACCPyr.swc')
    locs = [LOC_SOMA, LOC_DEND]
    # create a downsampled tree for efficiency
    m_tree.setCompTree()
    m_tree.distributeLocsUniform(50., add_bifurcations=True, name='downsampled')
    morph_tree = m_tree.createNewTree('downsampled', store_loc_inds=True)
    # get the reduced loc inds
    locinds = m_tree.getNearestLocinds(locs, name='downsampled')
    locinds_nt = [n.content['loc ind'] for n in morph_tree]
    nodeinds_newloc = [np.where(locinds_nt == loc_ind)[0][0] for loc_ind in locinds]
    nodes = [node for ii, node in enumerate(morph_tree) if ii in nodeinds_newloc]
    morph_locs = [(n.index, 1.) for n in nodes]

    return m_tree, morph_tree, locs, morph_locs
################################################################################


class DataContainer(object):
    """
    Class to conveniently acces the data
    """
    def __init__(self, with_zd=True):
        if with_zd:
            arr = np.fromfile('datafiles/Concatenated_ACC_Dzd.dat', sep=' ')
            self.t_d, self.v_d = arr[::2], arr[1::2]

            arr = np.fromfile('datafiles/Concatenated_ACC_Szd.dat', sep=' ')
            self.t_s, self.v_s = arr[::2], arr[1::2]

        else:
            arr = np.fromfile('datafiles/Concatenated_ACC_Dhcn.dat', sep=' ')
            self.t_d, self.v_d = arr[::2], arr[1::2]

            arr = np.fromfile('datafiles/Concatenated_ACC_Shcn.dat', sep=' ')
            self.t_s, self.v_s = arr[::2], arr[1::2]

        self.dt = self.t_d[1] - self.t_d[0]
        # cut out last part of trace
        i1 = int(4000./self.dt)
        self.t_d, self.v_d = self.t_d[:i1], self.v_d[:i1]
        self.t_s, self.v_s = self.t_s[:i1], self.v_s[:i1]
################################################################################