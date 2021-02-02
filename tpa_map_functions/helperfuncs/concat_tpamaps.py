import os.path
import numpy as np

"""
Created by: Dominik Staerk
Created on: 04.11.2020
"""


def concat_tpamaps(path2tpamaps: str(),
                   filepath2output: str(),
                   list_tpamaps: list(),
                   time_interpsteps: list(),
                   bool_enable_tpamaps: bool()):
    """Concatenates the generated tpamaps horizontally to further use them within the vehicle dynamics simulation.

    This is mandatory because the compiled friction_model in the vehicle_dynamics_model that processes the tpa-maps only
    can handle fixed size array inputs.
    The concatenated tpamap.csv will hold the tpamap_Mode in row 1 and column 1, the interpTime array in row 2 and
    column 1 to 10 and the map information in rows 3 to 2502 and column 1 to 23.
    A maximum number of 10 tpamaps can be concatenated and further processed!

    :param path2tpamaps: path to folder where separate tpamaps are located
    :type path2tpamaps: str
    :param filepath2output: path to file where concatenated tpamaps should get saved
    :type filepath2output: str
    :param list_tpamaps: list containing the filenames of the tpamap which should get concatenated
    :type list_tpamaps: list
    :param time_interpsteps: list containing the conrete timesteps used for interpolating the tpamaps
    :type time_interpsteps: list
    :param bool_enable_tpamaps: flag which enables or disables the tpamaps when the concatenated output file is loaded
    :type bool_enable_tpamaps: bool
    """

    # check arguments --------------------------------------------------------------------------------------------------

    if len(list_tpamaps) > 10:
        ValueError("tpa map functions: list 'list_tpamaps' must not exceed 10 elements")
    elif len(time_interpsteps) > 10:
        ValueError("tpa map functions: list 'time_interpsteps' must not exceed 10 elements")

    while len(list_tpamaps) < 10:
        list_tpamaps.append(False)

    while len(time_interpsteps) < 10:
        time_interpsteps.append(0.0)

    # initialize empty array to concatenate tpa maps
    tpamap = np.zeros((2502, 23))

    # load and concatenate maps
    k = 0
    for map in list_tpamaps:
        if map:
            filepath2input_tpamaps = os.path.join(path2tpamaps, map)

            with open(filepath2input_tpamaps, 'r') as fh:
                csv_data_tpamap = np.genfromtxt(fh, delimiter=',')
                tpamap_size = len(csv_data_tpamap)

        else:
            break

        tpamap[2:tpamap_size + 2, 0] = csv_data_tpamap[:, 0]
        tpamap[2:tpamap_size + 2, 1] = csv_data_tpamap[:, 1]
        tpamap[2:tpamap_size + 2, 2] = csv_data_tpamap[:, 2]
        tpamap[2:tpamap_size + 2, 3 + k * 2] = csv_data_tpamap[:, 3]
        tpamap[2:tpamap_size + 2, 4 + k * 2] = csv_data_tpamap[:, 4]
        k += 1

    # write data to tpamap_tum_mcs.csv
    if bool_enable_tpamaps:
        tpamap[0, 0] = 1.0
    else:
        tpamap[0, 0] = 0.0

    tpamap[1, :len(time_interpsteps)] = time_interpsteps[:]

    with open(filepath2output, 'wb') as fh:
        np.savetxt(fh, tpamap, fmt='%0.4f', delimiter=',')

    print('tpamap_tum_mcs.csv saved successfully')


if __name__ == "__main__":

    list_tpamaps = ['tpamap_berlin_1.csv',
                    'tpamap_berlin_2.csv',
                    'tpamap_berlin_3.csv',
                    'tpamap_berlin_4.csv']

    time_interpsteps = [0.0, 20.0, 35.0, 77.0]

    bool_enable_tpamaps = True

    path2tpamaps = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions', 'outputs')

    filepath2output = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0],
                                   'tpa_map_functions', 'outputs', 'tpamap_tum_mcs.csv')

    concat_tpamaps(path2tpamaps=path2tpamaps,
                   filepath2output=filepath2output,
                   list_tpamaps=list_tpamaps,
                   time_interpsteps=time_interpsteps,
                   bool_enable_tpamaps=bool_enable_tpamaps)
