import numpy as np
import logging
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

"""
Created by: Leonhard Hermansdorfer
Created on: 12.11.2019
"""


def preprocess_ltplrefline(filepath2ltpl_refline: str = str(),
                           reference_line: np.array = None,
                           stepsize_resample_m: float = 0,
                           interpolation_method: str = 'slinear',
                           logger: object = None) -> dict:
    """
    Documentation
    This function reads the reference line file of also used in local trajectory module to obtain reference line,
    track width, track boundaries and global raceline.
    If the reference line is already available, it can be used directly. If so, the filepath has to be a empty string!

    Input
    :param filepath2ltpl_refline:   path pointing to the file to be imported
    :param reference_line:          reference line containing xy-coordinates in meters [x_m, y_m]
    :param stepsize_resample_m:     desired stepsize for resampled reference line in meters
    :param interpolation_method:    interpolation method used for resampling of reference line
    :param logger:                  logger object for handling logs within this function

    Output
    :return refline:                x and y coordinate of refline
    :return width right/left:       witdth to track bounds at given refline coordinates in meters
    :return normvec_normalized:     x and y components of normized normal vector at given refline coordinates
    :return raceline_glob:          x,y-coordinates of global raceline
    :return bool_closedtrack:       boolean indicating whether race track is closed (True) or not (False)
    :return refline_resampled:      resampled reference line with reqeusted stepsize
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Import Data ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if not logger:
        logger = logging.getLogger('logs')

    dict_output = dict()

    # raise error, if both filepath and reference line array is provided
    if bool(filepath2ltpl_refline) and reference_line is not None:
        logger.critical('ERROR: path to reference line file AND reference line array provided! Only provide one input.')
        raise ValueError('path to reference line file AND reference line array provided!')

    # raise error, if neither filepath nor reference line array is provided
    elif not filepath2ltpl_refline and reference_line is None:
        logger.critical('ERROR: neither path to reference line file nor reference line array provided! At least '
                        'provide one input.')
        raise ValueError('neither path to reference line file nor reference line array provided!')

    # load reference line from file or proceed with existing reference line
    if bool(filepath2ltpl_refline):

        # load reference line
        with open(filepath2ltpl_refline, 'r') as fh:
            csv_data_refline = np.genfromtxt(fh, delimiter=';')

        # Parse csv file -----------------------------------------------------------------------------------------------

        # load data from csv file (closed; assumed order listed below)
        # x_ref_m, y_ref_m, width_right_m, width_left_m, x_normvec_m, y_normvec_m, alpha_m, s_racetraj_m,
        # psi_racetraj_rad, kappa_racetraj_radpm, vx_racetraj_mps, ax_racetraj_mps2

        # get reference line coordinates (x_ref_m, y_ref_m)
        refline_coordinates = csv_data_refline[:, 0:2]

        # get trackwidth right/left
        width_right = csv_data_refline[:, 2]
        width_left = csv_data_refline[:, 3]

        # get normized normal vectors
        normvec_normalized = csv_data_refline[:, 4:6]

        # get raceline alpha
        alpha_mincurv = csv_data_refline[:, 6]

        # get racline segment lengths
        length_rl = np.diff(csv_data_refline[:, 7])

        # get kappa at raceline points
        #kappa_rl = csv_data_refline[:, 9] # noqa F841

        # get velocity at raceline points
        #vel_rl = csv_data_refline[:, 10] # noqa F841

        # TESTING
        # refline_coordinates = refline_coordinates[0:500, :]
        # width_right = width_right[0:500]
        # width_left = width_left[0:500]
        # normvec_normalized = normvec_normalized[0:500, :]
        # alpha_mincurv = alpha_mincurv[0:500]
        # length_rl = length_rl[0:500]

        # Create output data -------------------------------------------------------------------------------------------

        # check whether race track is closed or not
        # calculate distance between first and last coordinate of reference line
        distance_last2firstcoordinate_m = math.sqrt(np.power(refline_coordinates[0, 0]
                                                             - refline_coordinates[-1, 0], 2)
                                                    + np.power(refline_coordinates[0, 1]
                                                               - refline_coordinates[-1, 1], 2))

        # calculate cumulative track length
        s = np.concatenate(([0], np.cumsum(length_rl)))
        # calculate coordinates of global raceline
        xy = refline_coordinates + normvec_normalized * alpha_mincurv[:, np.newaxis]

        # consider a reference line as not closed when distance between first and last entry is above 8 meters
        if distance_last2firstcoordinate_m < 8:
            raceline_glob = np.column_stack((s, xy))

        else:
            # add an additional entry at the end of each array (necessary for subsequent algorithms)
            diff_raceline_m = np.diff(xy[-2:, :], axis=0)[0]

            raceline_glob = np.column_stack((s, np.vstack([xy, xy[-1] + diff_raceline_m])))
            raceline_glob[-1, 0] = round(raceline_glob[-2, 0] + math.sqrt(np.sum(np.square(diff_raceline_m))), 7)

            width_right = np.hstack([width_right, width_right[-1]])
            width_left = np.hstack([width_left, width_left[-1]])

            normvec_normalized = np.vstack([normvec_normalized, normvec_normalized[-1, :]])

        dict_output = {'width_right': width_right,
                       'width_left': width_left,
                       'normvec_normalized': normvec_normalized,
                       'raceline_glob': raceline_glob}

    else:
        refline_coordinates = reference_line

    # check whether race track is closed or not
    # calculate distance between first and last coordinate of reference line
    distance_last2firstcoordinate_m = math.sqrt(np.power(refline_coordinates[0, 0] - refline_coordinates[-1, 0], 2)
                                                + np.power(refline_coordinates[0, 1] - refline_coordinates[-1, 1], 2))

    # consider a reference line as not closed when distance between first and last entry is above 8 meters
    if distance_last2firstcoordinate_m < 8:
        bool_closedtrack = True

        refline = refline_coordinates

    else:
        bool_closedtrack = False

        # add an additional entry at the end of each array (necessary for subsequent algorithms)
        diff_refline_m = np.diff(refline_coordinates[-2:, :], axis=0)[0]
        refline = np.vstack([refline_coordinates, refline_coordinates[-1] + diff_refline_m])

    dict_output['refline'] = refline
    dict_output['bool_closedtrack'] = bool_closedtrack

    # ------------------------------------------------------------------------------------------------------------------
    # Resample Reference Line ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if stepsize_resample_m >= 0.1:

        # testing
        # use reference line instead of raceline for further calculation
        s_refline_m = np.cumsum(np.sqrt(np.sum((np.square(np.diff(refline[:, 0])),
                                                np.square(np.diff(refline[:, 1]))), axis=0)))

        s_refline_m = np.vstack((np.zeros(1), s_refline_m[:, np.newaxis]))

        refline_concat = np.hstack((s_refline_m, refline))

        # check whether or not the provided stepsize is an even number; round to next int value if necessary
        if stepsize_resample_m % 1 != 0:
            logger.warning("resample stepsize has to be even! current value of stepsize_resample_m = "
                           + str(stepsize_resample_m) + " m; continue with " + str(round(stepsize_resample_m)) + " m")

            stepsize_resample_m = int(round(stepsize_resample_m))

        # check whether resampling is necessary or initial stepsize of raceline is already correct
        diff_coordinates_m = np.sqrt(np.sum(np.diff(refline_concat[:, 1:3], axis=0) ** 2, axis=1))

        if abs((max(abs(diff_coordinates_m)) - stepsize_resample_m)) <= 0.2:
            refline_resampled = refline_concat

        else:
            """ interpolation along a 2d curve
            source: https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
            """

            # coordinates to interpolate
            coordinates = refline_concat[:, 1:3]

            # linear length along the line:
            distance = refline_concat[:, 0] / refline_concat[-1, 0]

            steps = int(refline_concat[-1, 0] // stepsize_resample_m + 1)

            alpha = np.linspace(0, 1, steps)

            if interpolation_method == 'test':

                interpolated_points = {}

                # plot different interpolation methods for comparison
                # Interpolation for different methods:
                interpolations_methods = ['slinear', 'quadratic', 'cubic']
                for method in interpolations_methods:
                    interpolator = interp1d(distance, coordinates, kind=method, axis=0)
                    interpolated_points[method] = interpolator(alpha)

                # Graph:
                plt.figure(figsize=(7, 7))
                for method_name, curve in interpolated_points.items():
                    plt.plot(*curve.T, '-', label=method_name)

                plt.plot(refline_concat[:, 1], refline_concat[:, 2], 'ok', label='original points')
                plt.axis('equal')
                plt.legend()
                plt.xlabel('x in meters')
                plt.ylabel('y in meters')

                plt.show()

                return dict()

            else:
                interpolator = interp1d(distance, coordinates, kind=interpolation_method, axis=0)
                interpolated_points = interpolator(alpha)

            # calculate distance between resampled reference line coordinates
            diff_coordinates_m = np.sqrt(np.sum(np.diff(interpolated_points, axis=0) ** 2, axis=1))

            s = np.concatenate(([0], np.cumsum(diff_coordinates_m)))

            refline_resampled = np.column_stack((s, interpolated_points))

        mean_diff_m = np.mean(diff_coordinates_m)
        min_diff_m = np.min(diff_coordinates_m)
        max_diff_m = np.max(diff_coordinates_m)
        std_diff_m = np.std(diff_coordinates_m)

        logger.debug('mean distance between coordinates: ' + str(round(mean_diff_m, 3)) + ' m; '
                     + 'min. distance between coordinates: ' + str(round(min_diff_m, 3)) + ' m; '
                     + 'max. distance between coordinates: ' + str(round(max_diff_m, 3)) + ' m; '
                     + 'standard deviation of distance between coordinates: ' + str(round(std_diff_m, 3)) + ' m')

        dict_output['refline_resampled'] = {'refline_resampled': refline_resampled,
                                            'mean_diff_m': mean_diff_m,
                                            'min_diff_m': min_diff_m,
                                            'max_diff_m': max_diff_m,
                                            'std_diff_m': std_diff_m}

    return dict_output


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    import os.path
    import sys

    # import custom modules
    path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0],
                            'tpa_map_functions')

    sys.path.append(path2tmf)

    bool_plot = True

    filepath2ltpl_refline = os.path.join(path2tmf, 'example_files', 'traj_ltpl_cl', 'traj_ltpl_cl_berlin.csv')

    output_data = preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline,
                                         stepsize_resample_m=8.121358)

    if bool_plot and bool(output_data):

        refline_original = output_data['refline']
        refline_resampled = output_data['refline_resampled']['refline_resampled']

        plt.figure(figsize=(7, 7))

        plt.plot(refline_original[:, 0], refline_original[:, 1], 'k--', label='original reference line')
        plt.plot(refline_original[:, 0], refline_original[:, 1], 'kx', label='original reference line')
        plt.plot(refline_resampled[:, 1], refline_resampled[:, 2], 'r', label='resampled reference line')
        plt.plot(refline_resampled[:, 1], refline_resampled[:, 2], 'rx', label='resampled reference line')

        plt.axis('equal')
        plt.legend()
        plt.xlabel('x in meters')
        plt.ylabel('y in meters')

        plt.show()

        # plot histogram containing distances between coordinate points
        plt.figure()

        plt.hist(np.sqrt(np.sum(np.diff(refline_resampled[:, 1:3], axis=0) ** 2, axis=1)), bins=20)

        plt.axvline(x=output_data['refline_resampled']['mean_diff_m'], color='g', label='mean')
        plt.axvline(x=(output_data['refline_resampled']['mean_diff_m']
                       + output_data['refline_resampled']['std_diff_m']), color='y', label='stand.dev.')
        plt.axvline(x=(output_data['refline_resampled']['mean_diff_m']
                       - output_data['refline_resampled']['std_diff_m']), color='y')
        plt.axvline(x=output_data['refline_resampled']['min_diff_m'], color='r', label='min/max')
        plt.axvline(x=output_data['refline_resampled']['max_diff_m'], color='r')

        plt.legend()
        plt.grid()
        plt.xlabel('distance between reference line coordinate points in meters')
        plt.ylabel('bin count')

        plt.show()
