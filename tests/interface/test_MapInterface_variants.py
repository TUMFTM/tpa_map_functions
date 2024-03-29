import sys
import os
import numpy as np

# custom modules
path2module = os.path.join(os.path.abspath(__file__).split("tpa_map_functions")[0], "tpa_map_functions")

sys.path.append(path2module)

import tpa_map_functions.interface.MapInterface as MapInterface

"""
Created by: Leonhard Hermansdorfer
Created on: 10.12.2019
"""

# Get data from TPA
zmq_opts = {"ip": "localhost",          # IP of device running map interface
            "port_data": "47208",       # port number, standard: "47208"
            "topic": "tpa_to_ltpl"      # zmq topic of publisher
            }

# test 1 - constant local acceleration limits, const velocity
# test 2 - variable local acceleration limits, const velocity
# test 3 - constant local acceleration limits, variable velocity
# test 4 - variable local acceleration limits, variable velocity

filename_tpamap = ['tpamap_constloc_constvel.csv',
                   'tpamap_constloc_varvel.csv',
                   'tpamap_varloc_constvel_berlin.csv',
                   'tpamap_varloc_varvel_berlin.csv']

bool_enable_velocitydependence = [False,
                                  True,
                                  False,
                                  True]

# run all tests with interpolation disabled (=False) and enabled (True)
for ele in [False, True]:
    counter = 0

    while counter <= len(filename_tpamap) - 1:

        print('----------------------------------------------------------')
        print('run test {} with file: {}\nInterpolation: {}'.format(counter + 1, filename_tpamap[counter], ele))

        MapInt = MapInterface.MapInterface(filepath2localgg=os.path.join(path2module, 'inputs', 'veh_dyn_info',
                                                                         filename_tpamap[counter]),
                                           bool_enable_interpolation=ele,
                                           bool_enable_velocitydependence=bool_enable_velocitydependence[counter])

        pos_xy = np.random.rand(566, 2)
        pos_s = np.arange(0, 1501, 25)[:, np.newaxis]

        if bool_enable_velocitydependence[counter]:
            velocity_xy = np.random.rand(566, 1) * 100
            velocity_s = np.random.rand(pos_s.shape[0], pos_s.shape[1]) * 100
            velocity_emergency = np.arange(0, 100, 2.33)

        else:
            velocity_xy = np.asarray([])
            velocity_s = np.asarray([])
            velocity_emergency = np.asarray([])

        # xy-coordinates
        gg_xy = MapInt.get_acclim_tpainterface(position_m=pos_xy,
                                               position_mode='xy-cosy',
                                               velocity_mps=velocity_xy)

        # s-coordinates
        gg_s = MapInt.get_acclim_tpainterface(position_m=pos_s,
                                              position_mode='s-cosy',
                                              velocity_mps=velocity_s)

        acc_emergency = MapInt.get_acclim_tpainterface(position_m=np.asarray(0),
                                                       position_mode='emergency',
                                                       velocity_mps=velocity_emergency)

        if gg_xy.shape[1] != 2 and gg_s.shape[1] != 2:
            raise ValueError('TEST: TPA MapInterface: wrong shape of output local gg array!')

        del MapInt

        print('test {} passed!'.format(counter + 1))
        print('----------------------------------------------------------')

        counter += 1

print('tests passed')
