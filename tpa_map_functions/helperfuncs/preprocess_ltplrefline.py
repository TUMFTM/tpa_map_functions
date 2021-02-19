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
                           mode_resample_refline: str = 'const_steps',
                           stepsize_resample_m: float = 0,
                           interpolation_method: str = 'slinear',
                           logger: object = None,
                           bool_enable_debug: bool = False) -> dict:
    """
    Documentation
    This function reads the reference line file of also used in local trajectory module to obtain reference line,
    track width, track boundaries and global raceline.
    If the reference line is already available, it can be used directly. If so, the filepath has to be a empty string!

    Input
    :param filepath2ltpl_refline:   path pointing to the file to be imported
    :param reference_line:          reference line containing xy-coordinates in meters [x_m, y_m]
    :param mode_resample_refline:   mode for resampling reference line, options: "const_steps", "var_steps"
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
    :return bool_enable_debug:      enables debug mode and provides more data in output dictionary
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

    if reference_line is not None:
        mode_resample_refline = "const_steps"

    if mode_resample_refline not in ["const_steps", "var_steps"]:
        logger.critical('ERROR: provided mode for resampling reference line is not valid!')
        raise ValueError('provided mode for resampling reference line is not valid!')

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

        # get racline segments
        s_rl = csv_data_refline[:, 7]

        # get kappa at raceline points
        kappa_rl = csv_data_refline[:, 9]

        # get velocity at raceline points
        vel_rl = csv_data_refline[:, 10]

        # get long. acceleration at raceline points
        ax_rl = csv_data_refline[:, 11]

        # calculate lateral acceleration at raceline points
        ay_rl = kappa_rl * vel_rl**2

        # TESTING - test an unclosed race track
        # idx_cut = 333

        # refline_coordinates = refline_coordinates[0:idx_cut, :]
        # width_right = width_right[0:idx_cut]
        # width_left = width_left[0:idx_cut]
        # normvec_normalized = normvec_normalized[0:idx_cut, :]
        # alpha_mincurv = alpha_mincurv[0:idx_cut]
        # s_rl = s_rl[0:idx_cut]
        # kappa_rl = kappa_rl[0:idx_cut]
        # vel_rl = vel_rl[0:idx_cut]
        # ax_rl = ax_rl[0:idx_cut]
        # ay_rl = ay_rl[0:idx_cut]

        # calculate coordinates of raceline
        xy = refline_coordinates + normvec_normalized * alpha_mincurv[:, np.newaxis]

    else:
        refline_coordinates = reference_line

    # Check reference and race line ------------------------------------------------------------------------------------

    # calculate distance between first and last coordinate of reference line
    distance_last2firstcoordinate_m = \
        math.sqrt(np.power(refline_coordinates[0, 0] - refline_coordinates[-1, 0], 2)
                  + np.power(refline_coordinates[0, 1] - refline_coordinates[-1, 1], 2))

    # consider a reference line as not closed when distance between first and last entry is above 8 meters
    if distance_last2firstcoordinate_m < 8:
        bool_closedtrack = True
        refline = refline_coordinates

        if bool(filepath2ltpl_refline):
            raceline_glob = np.column_stack((s_rl, xy))

    else:
        bool_closedtrack = False

        # add an additional entry at the end of each array (necessary for subsequent steps)

        diff_refline_m = np.diff(refline_coordinates[-2:, :], axis=0)[0]
        refline = np.vstack([refline_coordinates, refline_coordinates[-1] + diff_refline_m])

        if bool(filepath2ltpl_refline):
            diff_raceline_m = np.diff(xy[-2:, :], axis=0)[0]

            raceline_glob = np.column_stack((np.vstack([s_rl[:, np.newaxis], 0]),
                                             np.vstack([xy, xy[-1] + diff_raceline_m])))

            raceline_glob[-1, 0] = round(raceline_glob[-2, 0] + math.sqrt(np.sum(np.square(diff_raceline_m))), 7)

            kappa_rl = np.hstack([kappa_rl, kappa_rl[-1]])
            vel_rl = np.hstack([vel_rl, vel_rl[-1]])
            ax_rl = np.hstack([ax_rl, ax_rl[-1]])
            ay_rl = np.hstack([ay_rl, ay_rl[-1]])

    if bool(filepath2ltpl_refline):
        dict_output = {'raceline_glob': raceline_glob,
                       'width_right': np.hstack([width_right, width_right[-1]]),
                       'width_left': np.hstack([width_left, width_left[-1]]),
                       'normvec_normalized': np.vstack([normvec_normalized, normvec_normalized[-1, :]])}

    # use reference line instead of raceline for further calculation
    s_refline_m = np.cumsum(np.sqrt(np.sum((np.square(np.diff(refline[:, 0])),
                                            np.square(np.diff(refline[:, 1]))), axis=0)))

    s_refline_m = np.vstack((np.zeros(1), s_refline_m[:, np.newaxis]))

    refline_concat = np.hstack((s_refline_m, refline))

    dict_output['refline'] = refline_concat
    dict_output['bool_closedtrack'] = bool_closedtrack

    # ------------------------------------------------------------------------------------------------------------------
    # Resample Reference Line ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    diff_coordinates_m = np.sqrt(np.sum(np.diff(refline_concat[:, 1:3], axis=0) ** 2, axis=1))

    dict_output['refline_resampled'] = dict()

    # resample reference line with constant step size ------------------------------------------------------------------
    if mode_resample_refline == "const_steps":

        if stepsize_resample_m <= 0.1:
            logger.warning("desired stepsize for reference line resampling is below threshold "
                           + "-> proceed without resampled reference line")

        else:

            # check whether or not the provided stepsize is an even number; round to next int value if necessary
            if stepsize_resample_m % 1 != 0:
                logger.warning("resample stepsize has to be even! current value of stepsize_resample_m = "
                               + str(stepsize_resample_m) + " m; continue with " + str(round(stepsize_resample_m))
                               + " m")

                stepsize_resample_m = int(round(stepsize_resample_m))

                if np.isclose(stepsize_resample_m, 0, 1e-08):
                    stepsize_resample_m += 1

            # check whether resampling is necessary or initial stepsize of raceline is already correct
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

    # resample reference line with variable step size on basis of raceline ---------------------------------------------
    elif mode_resample_refline == "var_steps":

        ax_trigger = [0] * ax_rl.shape[0]
        ay_trigger = [0] * ay_rl.shape[0]

        # detect situations where long. or lat. acceleration exceed a certain limit
        for int_row, ele_row in enumerate(np.hstack((ax_rl[:, np.newaxis], ay_rl[:, np.newaxis]))):

            if ele_row[0] < - 1.0:
                ax_trigger[int_row] = -1
            elif ele_row[0] > 1.0:
                ax_trigger[int_row] = 1

            if ele_row[1] < - 1.0:
                ay_trigger[int_row] = -1
            elif ele_row[1] > 1.0:
                ay_trigger[int_row] = 1

        ay_trigger = np.asarray(ay_trigger)
        ax_trigger = np.asarray(ax_trigger)

        # filter situations which occur only for a single data point
        # TODO: this could be extended to more than one data point, e.g. 2 or 3 data points or even based on distance
        for i_count in range(3, len(ay_trigger)):

            if ax_trigger[i_count] == ax_trigger[i_count - 2] and ax_trigger[i_count] != ax_trigger[i_count - 1]:
                ax_trigger[i_count - 1] = ax_trigger[i_count - 2]

            if ay_trigger[i_count] == ay_trigger[i_count - 2] and ay_trigger[i_count] != ay_trigger[i_count - 1]:
                ay_trigger[i_count - 1] = ay_trigger[i_count - 2]

        # identify specific driving situations to resample reference line
        indices = []
        list_section_category = []
        list_sectcat_sparse = []
        section_category_prev = 0
        section_length_current = 0.0

        section_length_max = 200
        section_length_min = 15

        # section_categories:
        #   1 - pure braking
        #   2 - comined braking and turn (negative: left, positive: right)
        #   3 - pure turn (negative: left, positive: right)
        #   4 - combined acceleration and turn (negative: left, positive: right)
        #   5 - pure acceleration
        #   6 - high speed straight line

        diff_coordinates_m_ext = np.hstack((diff_coordinates_m, diff_coordinates_m[-1]))

        for i_count in range(len(ay_trigger)):

            # pure braking
            if ay_trigger[i_count] == 0 and ax_trigger[i_count] == -1:
                section_category = 1

            # combined braking and turn
            elif ay_trigger[i_count] != 0 and ax_trigger[i_count] == -1:
                section_category = 2 * np.sign(ay_trigger[i_count])

            # pure turning
            elif ay_trigger[i_count] != 0 and ax_trigger[i_count] == 0:
                section_category = 3 * np.sign(ay_trigger[i_count])

            # combined acceleration and turn
            elif ay_trigger[i_count] != 0 and ax_trigger[i_count] == 1:
                section_category = 4 * np.sign(ay_trigger[i_count])

            # pure acceleration
            elif ay_trigger[i_count] == 0 and ax_trigger[i_count] == 1:
                section_category = 5

            # high speed straight line
            elif ay_trigger[i_count] == 0 and ax_trigger[i_count] == 0 and vel_rl[i_count] > 40:
                section_category = 6

            else:
                section_category = -100

            # only after first iteration: set value of previous section
            if section_category_prev == 0:
                section_category_prev = section_category

            # check whether or not category has changed to previous data point
            if section_category_prev == section_category:

                # check whether current section length of same category exceeds max. section length
                if section_length_current < section_length_max:
                    section_length_current += diff_coordinates_m_ext[i_count]

                else:
                    section_length_current = 0.0
                    indices.append(i_count)
                    list_sectcat_sparse.append(section_category)

            elif section_category_prev != section_category:

                # check whether current section length of same category already exceeds min. section length
                if section_length_current >= section_length_min:
                    section_length_current = 0.0
                    indices.append(i_count)
                    section_category_prev = section_category
                    list_sectcat_sparse.append(section_category)

                else:
                    section_length_current += diff_coordinates_m_ext[i_count]
                    section_category = section_category_prev

            list_section_category.append(section_category)

        indices.insert(0, 0)
        indices.append(refline_concat.shape[0] - 1)
        list_sectcat_sparse.insert(0, list_section_category[0])
        list_sectcat_sparse.append(100)

        # postprocess sections -----------------------------------------------------------------------------------------
        prev = list_sectcat_sparse[0]
        count = 1

        for i_count in range(1, len(list_sectcat_sparse)):

            if list_sectcat_sparse[i_count] == prev:
                count += 1

            elif count > 1 and (list_sectcat_sparse[i_count] != prev or i_count == len(list_sectcat_sparse) - 1):
                logger.debug("number of consecutive sections of type {}: {} sections".format(prev, count))
                logger.debug("sections start at {} m, end at {}".format(refline_concat[indices[i_count - count], 0],
                                                                        refline_concat[indices[i_count], 0]))

                interp = np.linspace(refline_concat[indices[i_count - count], 0],
                                     refline_concat[indices[i_count], 0],
                                     count + 1)

                # calculate indices which should be used for interpolated sections
                for j_count in range(1, len(interp)):
                    idx = np.argmin(np.abs(refline_concat[:, 0] - interp[j_count]))
                    indices[i_count - count + j_count] = idx

                count = 1
                prev = list_sectcat_sparse[i_count]

            else:
                count = 1
                prev = list_sectcat_sparse[i_count]

        refline_resampled = refline_concat[indices, :]

        diff_coordinates_m = np.sqrt(np.sum(np.diff(refline_resampled[:, 1:3], axis=0) ** 2, axis=1))

        if bool_enable_debug:
            dict_output['refline_resampled'].update({'ax_mps2': ax_rl,
                                                     'ay_mps2': ay_rl,
                                                     'ax_trigger': ax_trigger,
                                                     'ay_trigger': ay_trigger,
                                                     'list_section_category': list_section_category})

    if mode_resample_refline in ["const_steps", "var_steps"] and bool_enable_debug:

        mean_diff_m = np.mean(diff_coordinates_m)
        min_diff_m = np.min(diff_coordinates_m)
        max_diff_m = np.max(diff_coordinates_m)
        std_diff_m = np.std(diff_coordinates_m)

        logger.debug('mean distance between coordinates: ' + str(round(mean_diff_m, 3)) + ' m; '
                     + 'min. distance between coordinates: ' + str(round(min_diff_m, 3)) + ' m; '
                     + 'max. distance between coordinates: ' + str(round(max_diff_m, 3)) + ' m; '
                     + 'standard deviation of distance between coordinates: ' + str(round(std_diff_m, 3)) + ' m')

        dict_output['refline_resampled'].update({'refline_resampled': refline_resampled,
                                                 'mean_diff_m': mean_diff_m,
                                                 'min_diff_m': min_diff_m,
                                                 'max_diff_m': max_diff_m,
                                                 'std_diff_m': std_diff_m})

    return dict_output


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # Import necessary modules
    import os.path

    # User Input -------------------------------------------------------------------------------------------------------

    track_name = 'modena'
    bool_enable_debug = True

    mode_resample_refline = 'var_steps'
    stepsize_resample_m = 11.11

    test_source = 'path'  # or 'path'

    # Preprocess Reference Line ----------------------------------------------------------------------------------------

    path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')

    filepath2ltpl_refline = os.path.join(path2tmf, 'inputs', 'traj_ltpl_cl', 'traj_ltpl_cl_' + track_name + '.csv')

    if test_source == 'file':

        # load reference line
        with open(filepath2ltpl_refline, 'r') as fh:
            csv_data_refline = np.genfromtxt(fh, delimiter=';')

        reference_line = csv_data_refline[:, 0:2]

        mode_resample_refline = 'const_steps'

        output_data = preprocess_ltplrefline(reference_line=reference_line,
                                             stepsize_resample_m=stepsize_resample_m,
                                             bool_enable_debug=bool_enable_debug)
    else:

        output_data = preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline,
                                             mode_resample_refline=mode_resample_refline,
                                             stepsize_resample_m=stepsize_resample_m,
                                             bool_enable_debug=bool_enable_debug)

    if bool_enable_debug:

        refline_original = output_data['refline']
        refline_resampled = output_data['refline_resampled']['refline_resampled']

        plt.figure(figsize=(7, 7))

        plt.plot(refline_original[:, 1], refline_original[:, 2], 'k--', label='original reference line')
        plt.plot(refline_original[:, 1], refline_original[:, 2], 'kx', label='original reference line')
        plt.plot(refline_resampled[:, 1], refline_resampled[:, 2], 'r', label='resampled reference line')
        plt.plot(refline_resampled[:, 1], refline_resampled[:, 2], 'ro', label='resampled reference line')

        plt.axis('equal')
        plt.legend()
        plt.xlabel('x in meters')
        plt.ylabel('y in meters')

        plt.show()

        if mode_resample_refline == "const_steps":

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

        elif mode_resample_refline == "var_steps":

            plt.figure()

            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(refline_original[:, 0], output_data['refline_resampled']['ax_mps2'], label="long. acc.")
            ax1.plot(refline_original[:, 0], output_data['refline_resampled']['ay_mps2'], label="lat. acc.")

            for s in refline_resampled[:, 0]:
                plt.vlines(s, -10, 10, colors='k', linestyle='--')

            plt.grid()
            plt.xlabel("track position in m")
            plt.ylabel("long./lat. acc. in mps2")
            plt.legend()

            ax2 = plt.subplot(2, 1, 2, sharex=ax1)

            ax2.step(refline_original[:, 0], np.multiply(output_data['refline_resampled']['ax_trigger'], 0.9),
                     where='post', linewidth=2.0, label="trigger: long. acc.")
            ax2.step(refline_original[:, 0], np.multiply(output_data['refline_resampled']['ay_trigger'], 0.8),
                     where='post', linewidth=2.0, label="trigger: lat. acc.")

            ax2.step(refline_original[:, 0],
                     np.multiply(output_data['refline_resampled']['list_section_category'], 1.0), where='post',
                     linewidth=2.0, label="section type")

            for s in refline_resampled[:, 0]:
                plt.vlines(s, -7, 7, colors='k', linestyle='--')

            plt.ylim([-7, 7])

            plt.grid()
            plt.xlabel("track position in m")
            plt.ylabel("section type")

            plt.legend()
            plt.show()
