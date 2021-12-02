import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt

# import custom modules
path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions as tmf

# User Input -------------------------------------------------------------------------------------------------------

track_name = 'monteblanco'

# Preprocess Reference Line ----------------------------------------------------------------------------------------
filepath2ltpl_refline = os.path.join(path2tmf, 'inputs', 'traj_ltpl_cl', 'traj_ltpl_cl_' + track_name + '.csv')

# load reference line
with open(filepath2ltpl_refline, 'r') as fh:
    csv_data_refline = np.genfromtxt(fh, delimiter=';')

test_array = np.vstack((csv_data_refline, csv_data_refline))

position_samples = test_array[:, 0:2] + (1 - np.random.rand(test_array.shape[0], 2))

test = tmf.helperfuncs.transform_coordinates_xy2s. \
    transform_coordinates_xy2s(coordinates_sxy_m=np.hstack((csv_data_refline[:, 7][:, np.newaxis],
                                                            csv_data_refline[:, 0:2])),
                               position_m=position_samples,
                               s_tot_m=csv_data_refline[-1, 7])
