import numpy as np
import time
import sys
import os.path
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph

# custom modules
path2module = os.path.join(os.path.abspath(__file__).split("tpa_map_functions")[0], "tpa_map_functions")

sys.path.append(path2module)

import tpa_map_functions

"""
Created by: Leonhard Hermansdorfer
Created on: 10.12.2019
"""


# User Input -----------------------------------------------------------------------------------------------------------

trackname = 'IMS_2020_sim'
localgg_name = "tpamap_IMS_2020_sim_IMS"

updateFrequency = 100
laps = 1
s_terminate_m = 3000
bool_plot = True

# set indices for looping reference line during tpa map interface request

# length of trajectory for which tpa map info is requested during each step
idx_stop = 100

# index shift between each request (simulates a driving vehicle)
delta_idx = 80

# tpa interface settings
bool_enable_interface2tpa = False
bool_enable_interpolation = True
bool_enable_velocitydependence = False

# tpa zmq settings
zmq_opts = {"ip": "localhost",  # IP of device running map interface
            "port_data": "47208",  # port number, standard: "21212"
            "topic": "tpa_to_ltpl"  # zmq topic of publisher
            }


# Set Up ---------------------------------------------------------------------------------------------------------------

filepath2localgg = os.path.join(path2module, "inputs", "veh_dyn_info", localgg_name + ".csv")
filepath2ltpl_refline = os.path.join(path2module, 'inputs', 'traj_ltpl_cl', 'traj_ltpl_cl_' + trackname + '.csv')

# load reference line from file
dict_refline = tpa_map_functions.helperfuncs.preprocess_ltplrefline.\
    preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline)

refline = dict_refline['refline']
# s-coordinate of raceline (not used)
# s_m = dict_refline['raceline_glob'][:, 0]

# calculate s-coordinate of reference line (not raceline!)
length_s_m = np.hstack((0, np.cumsum(np.sqrt(np.sum(np.square(np.diff(refline, axis=0)), axis=1)))))

coordinates_sxy_m = np.hstack((length_s_m[:, np.newaxis], refline))

# create a map interface class
myInterface = tpa_map_functions.interface.MapInterface.\
    MapInterface(filepath2localgg=filepath2localgg,
                 zmq_opts_sub_tpa=zmq_opts,
                 bool_enable_interface2tpa=bool_enable_interface2tpa,
                 bool_enable_interpolation=bool_enable_interpolation,
                 bool_enable_velocitydependence=bool_enable_velocitydependence)

# testing --------------------------------------------------------------------------------------------------------------

lapcounter = 0
idx_start = 0
log_duration = []
output_data = []

while True:

    list_trajectories = []
    traj_scoord_m = []
    acc_lim = []

    if idx_stop > refline.shape[0] > idx_start:

        idx_stop -= refline.shape[0]
        trajectory = np.vstack((refline[idx_start:, :], refline[:idx_stop, :]))

    elif idx_stop > refline.shape[0]:

        idx_stop -= refline.shape[0]
        trajectory = refline[idx_start:idx_stop, :]

    elif idx_start > refline.shape[0]:

        idx_start -= refline.shape[0]
        trajectory = refline[idx_start:idx_stop, :]

        lapcounter += 1

        print('lap completed\n')

    elif idx_stop < idx_start:
        trajectory = np.vstack((refline[idx_start:, :], refline[:idx_stop, :]))

    else:
        trajectory = refline[idx_start:idx_stop, :]

    for row in trajectory:
        traj_scoord_m.append(tph.path_matching_global.path_matching_global(coordinates_sxy_m, row)[0])

    idx_start += delta_idx
    idx_stop += delta_idx

    # save start time
    t_start = time.perf_counter()

    acc_lim = myInterface.get_acclim_tpainterface(position_m=trajectory,
                                              position_mode='xy-cosy')

    myInterface.update()

    duration = time.perf_counter() - t_start
    sleep_time = 1 / updateFrequency - duration
    # print("sleep: {:.3f} s".format(sleep_time))
    # print("duration: {:.3f} s".format(duration))

    output_data.append(np.hstack((np.vstack(traj_scoord_m), acc_lim)))

    if sleep_time > 0.0:
        time.sleep(sleep_time)
    else:
        pass
        # logging.warning("Didn't get enough sleep...")

    log_duration.append(duration)

    if len(log_duration) == 100:
        print('mean duration over last 100 timesteps: ', np.mean(log_duration), ' s')
        print('max duration of last 100 timesteps: ', max(log_duration), ' s')
        print('min duration of last 100 timesteps: ', min(log_duration), ' s\n')

        log_duration = []

    if lapcounter == laps:
        break

    if s_terminate_m > 0 and traj_scoord_m[-1] > s_terminate_m:
        break

# plot results
if bool_plot:

    plt.figure()
    plt.xlim([0, length_s_m[-1]])

    for ele in output_data:
        plt.step(ele[:, 0], ele[:, 1])

        plt.draw()
        plt.pause(0.01)

    plt.step(myInterface.coordinates_sxy_m[:, 0],
             np.vstack((np.asarray(myInterface.localgg_mps2[0, 0]),
                        np.vstack(myInterface.localgg_mps2[:-1, 0]))),
             'k', label='ground truth of tpa map')

    plt.legend()
    plt.show()

    plt.figure()

    plt.step(myInterface.coordinates_sxy_m[:, 0],
             np.vstack((np.asarray(myInterface.localgg_mps2[0, 0]),
                        np.vstack(myInterface.localgg_mps2[:-1, 0]))),
             'k', label='ground truth of tpa map')

    plt.step(output_data[0][:, 0], output_data[0][:, 1])

    plt.show()
