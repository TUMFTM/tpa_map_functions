import os.path
import numpy as np
import datetime

"""
Created by: Leonhard Hermansdorfer
Created on: 04.02.2021
"""


def concat_tpamaps_ltpl(path2tpamaps: str(),
                        filepath2output: str(),
                        tpamap_identifier: str()):
    """Concatenates seperate tpamaps to a single, velocity-dependent tpamap which can be used for local trajectory
       planning.

        This function concatenates separate tpamaps to a single, velocity-dependent tpamap. The input files must have
        an identical identifier (prefix). The only difference comes from the distinct velocity steps where each tpamap
        is valid. This velocity info must be included in the filenames.
        Example: tpamap_berlin__27mps.csv, tpamap_berlin__56mps.csv, tpamap_berlin__83mps.csv

    :param path2tpamaps: path to folder where seperate tpamaps are located.
    :type path2tpamaps: str
    :param filepath2output: path to file where the concatenated tpamap should get stored.
    :type filepath2output: str
    :param tpamap_identifier: Identifier/ identical part of name of separate tpamap which should get concatenated.
    :type tpamap_identifier: str
    """

    i_count_files = 0
    list_velsteps = []
    list_tpamap_filenames = []

    # list all files in specified input folder
    for file in os.listdir(path2tpamaps):

        # search for files which end with specific extention and contain identifier
        if file.endswith(".csv") and tpamap_identifier in file:

            if "mps" in file:
                list_velsteps.append(int((file.split('_')[-1]).split('mps')[0]))

            list_tpamap_filenames.append(file)

            i_count_files += 1

    # sort filenames and velocity steps -> increasing order
    list_tpamap_filenames.sort()
    list_velsteps.sort()

    for j_count in range(len(list_velsteps)):
        list_dummy = list_velsteps.copy()

        val = list_dummy.pop(j_count)

        if val in list_dummy:
            raise ValueError("tpamap functions: velocity steps of to-be-concatenated tpamaps must be not equal!")

    # load reference line
    with open(os.path.join(path2tpamaps, list_tpamap_filenames[0]), 'r') as fh:
        csv_data_tpamap = np.genfromtxt(fh, delimiter=',', comments='#')
        tpamap_size = len(csv_data_tpamap)

    tpamap = np.zeros((tpamap_size, i_count_files * 2 + 4))

    # load and concatenate maps
    i_count = 0
    for filename in list_tpamap_filenames:

        # load reference line
        with open(os.path.join(path2tpamaps, filename), 'r') as fh:
            csv_data_tpamap = np.genfromtxt(fh, delimiter=',', comments='#')
            tpamap_size = len(csv_data_tpamap)

        if i_count == 0:
            tpamap[:, 0] = csv_data_tpamap[:, 0]
            tpamap[:, 1] = csv_data_tpamap[:, 1]
            tpamap[:, 2] = csv_data_tpamap[:, 2]
            tpamap[:, 3] = csv_data_tpamap[:, 3]

        else:

            if not np.all(np.equal(tpamap[:, :4], csv_data_tpamap[:, :4])):
                raise ValueError("tpamap functions: sxy-coordinates of included tpamaps are not equal!")

        tpamap[:, 4 + i_count * 2] = csv_data_tpamap[:, 4]
        tpamap[:, 5 + i_count * 2] = csv_data_tpamap[:, 5]
        i_count += 1

    # TODO: plausibility checks
    #       - check whether every acc. limit is not below value at smaller velocity level

    # save data to csv file --------------------------------------------------------------------------------------------

    # prepare file header
    header = 'created on: ' + datetime.datetime.now().strftime("%Y-%m-%d") + ', '\
             + datetime.datetime.now().strftime("%H:%M:%S")

    header = header + '\n' + 'track: ' + tpamap_identifier
    header = header + '\n' + 'section_id,s_m,x_m,y_m'

    if len(list_velsteps) <= 1:
        header = header + ',ax_max_mps2,ay_max_mps2'

    else:
        for velstep in list_velsteps:
            header = header + ',ax_max_mps2__' + str(velstep) + 'mps' + ',ay_max_mps2__' + str(velstep) + 'mps'

    # write data file
    with open(filepath2output, 'wb') as fh:
        np.savetxt(fh, tpamap, fmt='%0.4f', delimiter=',', header=header)

    print('tpa map functions: tpamap saved successfully')


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
