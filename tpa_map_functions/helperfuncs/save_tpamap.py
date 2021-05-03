import numpy as np
import os.path
import datetime

"""
Created by: Leonhard Hermansdorfer
Created on: 08.01.2020
"""


def save_tpamap_fromfile(filepath2output_tpamap: str,
                         mode_save_tpamap: str,
                         coordinates_sxy_m: np.array,
                         long_limit: np.array,
                         lat_limit: np.array,
                         track_name: str,
                         section_id: np.array = None,
                         header_info: dict = dict()):
    """Creates a default tpa map and calls a separate function to save the generated tpa map to a csv-file.

    Input
    :param filepath2output_tpamap:      filepath to where tpa map is saved
    :type filepath2output_tpamap:       str
    :param mode_save_tpamap:            determines which data has to be saved (acc. limits or friction coeff.)
    :type mode_save_tpamap:             str
    :param coordinates_sxy_m:           contains s-,x- and y-coordinate of reference line
    :type coordinates_sxy_m:            np.array
    :param long_limit:                  value for initialization of long. acceleration limit of tpa map
    :type long_limit:                   np.array
    :param lat_limit:                   value for initialization of lat. acceleration limit of tpa map
    :type lat_limit:                    np.array
    :param section_id:                  section id for every coordinate
    :type section_id:                   np.array
    :param header_info:                 contains information which is placed into the file header, defaults to dict()
    :type header_info:                  dict, optional

    Output
    ---
    """

    len_refline = coordinates_sxy_m.shape[0]

    try:
        if long_limit.ndim == 0:

            if lat_limit.ndim == 0:
                long_limit = np.full((len_refline, 1), long_limit)
                lat_limit = np.full((len_refline, 1), lat_limit)

            elif lat_limit.shape[0] == len_refline:
                long_limit = np.full((len_refline, 1), long_limit)

        elif long_limit.shape[0] == len_refline:

            if lat_limit.ndim == 0:
                lat_limit = np.full((len_refline, 1), lat_limit)

    except ValueError:
        print('handling of ax and ay limits did fail - please check input again - data not saved!')
        return

    if mode_save_tpamap == 'acclimits':

        # calc default section_id when not provided via function
        if section_id is None:

            section_id = np.arange(1, len_refline + 1)[:, np.newaxis]

        # recalc acc. values to fill reference line
        elif section_id[0, 0] > 1:

            long_limit_ext = np.zeros((len_refline, 1))
            lat_limit_ext = np.zeros((len_refline, 1))
            long_limit_ext[-1] = long_limit[0]
            lat_limit_ext[-1] = lat_limit[0]

            sectionid_change = np.where(
                np.vstack((np.asarray([True]), np.abs(np.diff(section_id, axis=0)) > 0.5)))[0]

            for i_count, idx in enumerate(sectionid_change[0:-1]):
                long_limit_ext[idx:sectionid_change[i_count + 1]] = long_limit[i_count]
                lat_limit_ext[idx:sectionid_change[i_count + 1]] = lat_limit[i_count]

            long_limit = long_limit_ext
            lat_limit = lat_limit_ext

        data_output = np.hstack((section_id, coordinates_sxy_m, long_limit, lat_limit))

    # fill data array with friction coeffs
    elif mode_save_tpamap == 'frictioncoeff':
        data_output = np.hstack((coordinates_sxy_m, long_limit, lat_limit))

    else:
        raise ValueError('mode_save_tpamap unknown!')

    save_tpamap(filepath2output_tpamap=filepath2output_tpamap,
                tpamap=data_output,
                header_info=header_info,
                track_name=track_name)

# ----------------------------------------------------------------------------------------------------------------------


def save_tpamap(filepath2output_tpamap: str,
                tpamap: np.array,
                track_name: str,
                header_info: dict = dict()):

    """Saves the tpa map containing s-coordinate, x,y-coordinates and long./lat. acceleration limits to a csv-file.

    Input
    :param filepath2output_tpamap: filepath to where tpa map is saved
    :type filepath2output_tpamap: str
    :param tpamap: contains the entire tpa map data (s-,x-,y-coordinates, long. acc. limit, lat. acc. limit)
    :type tpamap: np.array
    :param header_info: contains information which is placed into the file header, defaults to dict()
    :type header_info: dict, optional

    Output
    ---
    """

    header = 'created on: ' + datetime.datetime.now().strftime("%Y-%m-%d") + ', '\
             + datetime.datetime.now().strftime("%H:%M:%S")

    try:
        header = header + '\n' + 'track: ' + track_name + '\n' + 'GUI mode: ' + str(header_info['gui_mode'])
    except KeyError:
        pass

    if header_info['gui_mode'] == 2:
        header = header + '\n' + 'section_id,s_m,x_m,y_m,ax_max_mps2,ay_max_mps2'
    else:
        header = header + '\n' + 's_m,x_m,y_m,lambda_mue_x,lambda_mue_y'

    with open(filepath2output_tpamap, 'wb') as fh:
        np.savetxt(fh, tpamap, fmt='%0.4f', delimiter=',', header=header)

        print('tpamap_' + track_name + ' saved successfully')


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    import sys

    # import custom modules
    path2module = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')

    sys.path.append(path2module)

    import tpa_map_functions

    bool_plot = True
    stepsize_resample_m = 10
    ax_max_tires_mps2 = np.asarray(10.5)
    ay_max_tires_mps2 = np.asarray(10)

    track_name = "dummy track"
    header_custom = {"gui_mode": 2}

    filepath2ltpl_refline = os.path.join(path2module, 'inputs', 'traj_ltpl_cl', 'traj_ltpl_cl_berlin.csv')
    filepath2output_tpamap = os.path.join(path2module, 'outputs', 'testmap.csv')

    dict_output = tpa_map_functions.helperfuncs.preprocess_ltplrefline.\
        preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline,
                               stepsize_resample_m=stepsize_resample_m)

    refline_resampled = dict_output['refline_resampled']['refline_resampled']

    # testing
    ax_max_tires_mps2 = np.random.normal(loc=ax_max_tires_mps2, size=(refline_resampled.shape[0], 1))
    ay_max_tires_mps2 = np.random.normal(loc=ay_max_tires_mps2, size=(refline_resampled.shape[0], 1))
    # testing end

    # ax_max_tires_mps2 = np.asarray(ax_max_tires_mps2)
    # ay_max_tires_mps2 = np.asarray(ay_max_tires_mps2)

    save_tpamap_fromfile(filepath2output_tpamap=filepath2output_tpamap,
                         mode_save_tpamap='acclimits',
                         coordinates_sxy_m=refline_resampled,
                         long_limit=ax_max_tires_mps2,
                         lat_limit=ay_max_tires_mps2,
                         track_name=track_name,
                         header_info=header_custom)
