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

trackname = 'berlin'
tpamap_name = "tpamap_varloc_varvel_berlin"

updateFrequency = 100
laps = 1
s_terminate_m = 3000
bool_plot = True

velocity_mps = 20

# set indices for looping reference line during tpa map interface request

# length of trajectory for which tpa map info is requested during each step
idx_stop = 100

# index shift between each request (simulates a driving vehicle)
delta_idx = 80

# tpa interface settings
bool_enable_interface2tpa = False
bool_enable_interpolation = True
bool_enable_velocitydependence = True

# tpa zmq settings
zmq_opts = {"ip": "localhost",  # IP of device running map interface
            "port_data": "47208",  # port number, standard: "21212"
            "topic": "tpa_to_ltpl"  # zmq topic of publisher
            }

# Set Up ---------------------------------------------------------------------------------------------------------------

filepath2tpamap = os.path.join(path2module, "inputs", "veh_dyn_info", tpamap_name + ".csv")
filepath2ltpl_refline = os.path.join(path2module, 'inputs', 'traj_ltpl_cl', 'traj_ltpl_cl_' + trackname + '.csv')

# load reference line from file
dict_refline = tpa_map_functions.helperfuncs.preprocess_ltplrefline.\
    preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline)

coordinates_sxy_m = dict_refline['refline']
refline = dict_refline['refline'][:, 1:3]

# create a map interface class
myInterface = tpa_map_functions.interface.MapInterface.\
    MapInterface(filepath2localgg=filepath2tpamap,
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

    if bool_enable_velocitydependence:
        arr_velocity_mps = np.full(trajectory.shape, velocity_mps)

    # save start time
    t_start = time.perf_counter()

    if bool_enable_velocitydependence:
        acc_lim = myInterface.get_acclim_tpainterface(position_m=trajectory,
                                                      position_mode='xy-cosy',
                                                      velocity_mps=arr_velocity_mps)

        acc_emergency = myInterface.get_acclim_tpainterface(position_m=np.asarray(0),
                                                            position_mode='emergency',
                                                            velocity_mps=np.arange(0, 100, 10))

    else:
        acc_lim = myInterface.get_acclim_tpainterface(position_m=trajectory,
                                                      position_mode='xy-cosy')

        acc_emergency = myInterface.get_acclim_tpainterface(position_m=np.asarray(0),
                                                            position_mode='emergency')

    myInterface.update()

    # simulate race strategy intervention
    # if traj_scoord_m[-1] > 500 and traj_scoord_m[-1] < 1200:
    #     myInterface.set_acclim_strategy(10, 8, True)

    # elif traj_scoord_m[-1] > 1200:
    #     myInterface.set_acclim_strategy(1, 1, False)

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

list_linestyle = ['-', '--', '-.', ':'] * 2
list_linestyle.sort()
list_linecolor = ['black', 'blue']

if bool_plot:

    plt.figure()
    plt.xlim([0, coordinates_sxy_m[-1, 0]])

    for i_count in range(int(myInterface.localgg_mps2.shape[1] / 2)):

        if int(myInterface.localgg_mps2.shape[1] / 2) > 1:
            label = 'ground truth of tpamap at ' + str(myInterface.velocity_steps[i_count + 1]) + ' mps'
        else:
            label = 'ground truth of tpamap'

        plt.step(myInterface.coordinates_sxy_m[:, 0],
                 np.vstack((np.asarray(myInterface.localgg_mps2[0, 0 + i_count * 2]),
                            np.vstack(myInterface.localgg_mps2[:-1, 0 + i_count * 2]))),
                 color=list_linecolor[i_count % 2], linestyle=list_linestyle[i_count], linewidth=2.0, label=label)

    for ele in output_data:
        plt.step(ele[:, 0], ele[:, 1])

        plt.draw()
        plt.pause(0.01)

    plt.grid()
    plt.legend()
    plt.show()

    # plt.figure()

    # plt.step(myInterface.coordinates_sxy_m[:, 0],
    #          np.vstack((np.asarray(myInterface.localgg_mps2[0, 0]),
    #                     np.vstack(myInterface.localgg_mps2[:-1, 0]))),
    #          'k', label='ground truth of tpa map')

    # plt.step(output_data[0][:, 0], output_data[0][:, 1])

    # plt.show()
