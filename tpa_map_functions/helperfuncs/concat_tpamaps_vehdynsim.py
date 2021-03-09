import os.path
import numpy as np

"""
Created by: Dominik Staerk
Created on: 04.11.2020
"""


def concat_tpamaps_vehdynsim(path2tpamaps: str(),
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

    # initialize empty array to concatenate tpa maps
    tpamap = np.zeros((2502, 23))

    if bool_enable_tpamaps:

        print('tpa map functions: bool_enable_tpamaps is True -> specified tpa maps will be concatenated')

        if len(list_tpamaps) > 10 or len(list_tpamaps) == 0:
            raise ValueError("tpa map functions: list 'list_tpamaps' must contain between one (min) and ten (max) tpa "
                             "maps")

        if len(time_interpsteps) > 10 or len(time_interpsteps) == 0:
            raise ValueError("tpa map functions: list 'time_interpsteps' must contain between one (min) and ten (max) "
                             "values")

        if len(list_tpamaps) != len(time_interpsteps):
            raise ValueError("tpa map functions: both lists 'list_tpamaps' and 'time_interpsteps' must have same "
                             "number of entries")

        if not np.isclose(a=time_interpsteps[0], b=0.0, atol=1e-7):
            time_interpsteps[0] = 0.0
            print('WARNING tpa map functions: time_interpsteps first entry is not zero, but will be set to zero')

        if not np.all(np.diff(time_interpsteps) > 0):
            raise ValueError("tpa map functions: list 'time_interpsteps' must only contain an increasing set of values")

        while len(list_tpamaps) < 10:
            list_tpamaps.append(False)

        while len(time_interpsteps) < 10:
            time_interpsteps.append(0.0)

        # insert flag and timesteps into tpamap output array
        tpamap[0, 0] = 1.0
        tpamap[1, :len(time_interpsteps)] = time_interpsteps[:]

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

    else:

        print('tpa map functions: bool_enable_tpamaps is False -> constant friction coeffs. will be applied')

        tpamap[0, 0] = 0.0
        tpamap[1, :10] = np.arange(0, 100, 10)
        tpamap[2:, :] = 1.0
        tpamap[2, 0] = 0.0

    # write data to tpamap_tum_mcs.csv
    with open(filepath2output, 'wb') as fh:
        np.savetxt(fh, tpamap, fmt='%0.4f', delimiter=',')

    print('tpa map functions: tpamap_tum_mcs.csv saved successfully')


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
