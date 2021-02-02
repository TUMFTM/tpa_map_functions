import sys
import os
import time

# custom modules
path_root2Module = os.path.join(os.path.abspath(__file__).split("tpa_map_functions")[0],
                                "tpa_map_functions")

sys.path.append(path_root2Module)

import tpa_map_functions.interface.MapInterface as MapInterface

"""
Created by: Leonhard Hermansdorfer
Created on: 10.12.2019
"""

filepath2localgg = os.path.join(path_root2Module, 'inputs', 'veh_dyn_info', 'localgg_constloc_constvel.csv')

# Get data from TPA
zmq_opts = {"ip": "localhost",          # IP of device running map interface
            "port_data": "47208",       # port number, standard: "47208"
            "topic": "tpa_to_ltpl"      # zmq topic of publisher
            }

# create a map interface class and kick it every 100ms
MapInt = MapInterface.MapInterface(filepath2localgg=filepath2localgg,
                                   zmq_opts_sub_tpa=zmq_opts,
                                   bool_enable_interface2tpa=True)

updateFrequency = 20

while True:

    # save start time
    t_start = time.perf_counter()

    # update
    MapInt.update()

    duration = time.perf_counter() - t_start
    sleep_time = 1 / updateFrequency - duration

    if sleep_time > 0.0:
        time.sleep(sleep_time)
    else:
        print("Didn't get enough sleep... (TPA Map Interface)")
