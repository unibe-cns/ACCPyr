import numpy as np

import optimizer
import plot_optimization_results as por
import plot_h_analysis as pha


## Example on optimizing a configuration #######################################
# optimizer.optimizeModel(channel_names=['L'], zd=False, suffix='_test')
################################################################################


# plot trace example without zd ################################################
# por.plotTrace(['L', 'K_ir', 'K_m35', 'h_u'], False)
################################################################################


# plot trace example with zd ###################################################
# por.plotTrace(['L', 'K_ir', 'K_m35'], True)
################################################################################


# plot errors of each configuration ############################################
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
# por.plotErrors(channel_names_list, zd_list)
################################################################################


## h-current analysis ##########################################################
hca = pha.HCurrentAnalyzer()
hca.plotAttenuation()
hca.plotLocalNMDA(n_syns=np.arange(1,25))
hca.plotNMDAInteraction(n_syns=np.arange(1,25))
################################################################################