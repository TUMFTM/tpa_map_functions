import numpy as np
import logging
import math

"""
Created by: Leonhard Hermansdorfer
Created on: 12.11.2019
"""


def preprocess_ltplrefline(filepath2ltpl_refline: str = str(),
                           reference_line: np.array = None,
                           mode_resample_refline: str = 'const_steps',
                           stepsize_resample_m: float = 0,
                           section_length_limits_m: np.array = None,
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
    :param section_length_limits:   desired min. and max. sections lenghts when variable steps are activated
    :param interpolation_method:    interpolation method used for resampling of reference line
    :param logger:                  logger object for handling logs within this function
    :param bool_enable_debug:       enables debug mode and provides more data in output dictionary

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

    if reference_line is not None:
        mode_resample_refline = "const_steps"

    if mode_resample_refline not in ["const_steps", "var_steps"]:
        logger.critical('ERROR: provided mode for resampling reference line is not valid!')
        raise ValueError('provided mode for resampling reference line is not valid!')

    if section_length_limits_m is None and mode_resample_refline == "var_steps":
        mode_resample_refline = "const_steps"
        logger.warning('WARNING: resampling mode is set to constant steps, '
                       'because no min/max section lengths are provided')

    if section_length_limits_m is not None and (np.any(np.less_equal(section_length_limits_m, 0))
                                                or (section_length_limits_m[0] >= section_length_limits_m[1])):
        logger.critical('ERROR: provided section length limits are not valid!')
        raise ValueError('provided section length limits are not valid!')

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

        # TESTING ------------------------------------------------------------------------------------------------------
        # test an unclosed race track
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
        # TESTING End --------------------------------------------------------------------------------------------------

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

            width_right = np.hstack([width_right, width_right[-1]])
            width_left = np.hstack([width_left, width_left[-1]])
            normvec_normalized = np.vstack([normvec_normalized, normvec_normalized[-1, :]])
            kappa_rl = np.hstack([kappa_rl, kappa_rl[-1]])
            vel_rl = np.hstack([vel_rl, vel_rl[-1]])
            ax_rl = np.hstack([ax_rl, ax_rl[-1]])
            ay_rl = np.hstack([ay_rl, ay_rl[-1]])

    if bool(filepath2ltpl_refline):
        dict_output = {'raceline_glob': raceline_glob,
                       'width_right': width_right,
                       'width_left': width_left,
                       'normvec_normalized': normvec_normalized}

    # use reference line instead of raceline for further calculation
    diff_coordinates_m = np.sqrt(np.sum(np.diff(refline, axis=0) ** 2, axis=1))

    s_refline_m = np.cumsum(diff_coordinates_m)
    s_refline_m = np.vstack((np.zeros(1), s_refline_m[:, np.newaxis]))

    refline_sxy = np.hstack((s_refline_m, refline))

    dict_output['refline'] = refline_sxy
    dict_output['bool_closedtrack'] = bool_closedtrack

    # ------------------------------------------------------------------------------------------------------------------
    # Resample Reference Line ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    dict_output['refline_resampled'] = dict()
    dict_output['refline_resampled'].update({'refline_resampled': refline_sxy})

    # resample reference line with constant step size ------------------------------------------------------------------
    if mode_resample_refline == "const_steps":

        str_log = str(stepsize_resample_m)
        diff_coordinates_m_mean = np.mean(diff_coordinates_m)

        # enforce usage of min. stepsize if desired stepsize is smaller
        if stepsize_resample_m < diff_coordinates_m_mean:
            stepsize_resample_m = int(np.round(diff_coordinates_m_mean))

        count = 1
        idx_prev = 0
        section_id = np.zeros(refline_sxy.shape[0])

        # assign same section id to every s-coordinate which lies between start and end coordinate of specific section
        while stepsize_resample_m * count < refline_sxy[-1, 0]:

            idx_tmp = np.argmin(np.abs(refline_sxy[:, 0] - stepsize_resample_m * count))
            section_id[idx_prev:idx_tmp] = count
            idx_prev = idx_tmp
            count += 1

        # fill last section entries which are smaller than required stepsize
        if idx_tmp < refline_sxy.shape[0]:
            section_id[idx_tmp:] = count

        # last section id entry has to be a new value
        section_id[-1] = section_id[-2] + 1

        # calc min/mean/max values for debugging
        print_info = refline_sxy[np.concatenate((np.asarray([True]), np.isclose(np.diff(section_id), 1, 1e-08))), 0]

        logger.warning("resample stepsize has to match stepsize of given reference line! "
                       + "desired stepsize = " + str_log + " m; "
                       + "continue with a min/mean/max stepsize of "
                       + str(np.around(np.min(np.diff(print_info)), 3)) + '/'
                       + str(np.around(np.mean(np.diff(print_info)), 3)) + '/'
                       + str(np.around(np.max(np.diff(print_info)), 3)) + " m")

        dict_output['refline_resampled']['section_id'] = section_id[:, np.newaxis]

    # resample reference line with variable step size on basis of raceline ---------------------------------------------
    if mode_resample_refline == "var_steps":

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

        # filter situations which occur only for a predefined number of data points (to avoid single outliers)

        # number of data points below which trigger points are "smoothed"
        delay_axtrigger = 3
        delay_aytrigger = 3

        count = [1, 1]
        prev = [0, 0]

        for i_count in range(len(ax_trigger)):

            if i_count == 0:
                prev = [ax_trigger[i_count], ay_trigger[i_count]]

            # check whether or not category has changed to previous data point
            bool_expr = np.equal(prev, [ax_trigger[i_count], ay_trigger[i_count]])
            if np.any(bool_expr):
                count += 1 * bool_expr

            bool_expr = np.logical_not(bool_expr)

            if np.any(bool_expr):

                # check whether current number of trigger points is below defined limit
                if np.any(count < max(delay_axtrigger, delay_aytrigger)):

                    # overwrite ax trigger points
                    if count[0] < delay_axtrigger and bool_expr[0]:
                        logger.debug("ax section at {} m is only {} steps long".format(s_refline_m[i_count - count[0]],
                                                                                       count[0]))
                        logger.debug("insert ax trigger: {}".format(ax_trigger[i_count - count[0] - 1]))

                        ax_trigger[i_count - count[0]:i_count] = ax_trigger[i_count - count[0] - 1]
                        count[0] = 1

                    # overwrite ay trigger points
                    if count[1] < delay_aytrigger and bool_expr[1]:
                        logger.debug("ay section at {} m is only {} steps long".format(s_refline_m[i_count - count[1]],
                                                                                       count[1]))
                        logger.debug("insert ay trigger: {}".format(ay_trigger[i_count - count[1] - 1]))

                        ay_trigger[i_count - count[1]:i_count] = ay_trigger[i_count - count[1] - 1]
                        count[1] = 1

                # reset counter if number of trigger points of one section is above defined limit
                if np.logical_and(count[0] >= delay_axtrigger, bool_expr[0]):
                    count[0] = 1
                if np.logical_and(count[1] >= delay_aytrigger, bool_expr[1]):
                    count[1] = 1

            prev[0] = ax_trigger[i_count]
            prev[1] = ay_trigger[i_count]

        # identify specific driving situations to resample reference line
        indices = []
        list_section_category = []
        list_sectcat_sparse = []
        section_category_prev = 0
        section_length_current = 0.0

        section_length_min = section_length_limits_m[0]
        section_length_max = section_length_limits_m[1]

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
        indices.append(refline_sxy.shape[0] - 1)
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
                logger.debug("sections start at {} m, end at {}".format(refline_sxy[indices[i_count - count], 0],
                                                                        refline_sxy[indices[i_count], 0]))

                interp = np.linspace(refline_sxy[indices[i_count - count], 0],
                                     refline_sxy[indices[i_count], 0],
                                     count + 1)

                # calculate indices which should be used for interpolated sections
                for j_count in range(1, len(interp)):
                    idx = np.argmin(np.abs(refline_sxy[:, 0] - interp[j_count]))
                    indices[i_count - count + j_count] = idx

                count = 1
                prev = list_sectcat_sparse[i_count]

            else:
                count = 1
                prev = list_sectcat_sparse[i_count]

        # new
        section_id = np.zeros(refline_sxy.shape[0], dtype=int)

        for idx in np.arange(len(indices) - 1):
            section_id[indices[idx]:indices[idx + 1]] = idx + 1

        section_id[-1] = section_id[-2] + 1

        dict_output['refline_resampled']['section_id'] \
            = ((section_id * 10 + np.abs(list_section_category)) * np.sign(list_section_category))[:, np.newaxis]

        # calculate data for debug plots
        if bool_enable_debug:
            dict_output['refline_resampled'].update({'ax_mps2': ax_rl,
                                                     'ay_mps2': ay_rl,
                                                     'ax_trigger': ax_trigger,
                                                     'ay_trigger': ay_trigger,
                                                     'list_section_category': list_section_category})

    dict_output['refline_resampled']['sectionid_change'] \
        = np.concatenate((np.asarray([True]), np.isclose(np.diff(section_id), 1, 1e-08)))

    if mode_resample_refline in ["const_steps", "var_steps"] and bool_enable_debug:

        diff_refline_resampled_m = np.diff(refline_sxy[dict_output['refline_resampled']['sectionid_change'], 0])

        mean_diff_m = np.mean(diff_refline_resampled_m)
        min_diff_m = np.min(diff_refline_resampled_m)
        max_diff_m = np.max(diff_refline_resampled_m)
        std_diff_m = np.std(diff_refline_resampled_m)

        logger.debug('mean distance between coordinates: ' + str(round(mean_diff_m, 3)) + ' m; '
                     + 'min. distance between coordinates: ' + str(round(min_diff_m, 3)) + ' m; '
                     + 'max. distance between coordinates: ' + str(round(max_diff_m, 3)) + ' m; '
                     + 'standard deviation of distance between coordinates: ' + str(round(std_diff_m, 3)) + ' m')

        dict_output['refline_resampled'].update({'mean_diff_m': mean_diff_m,
                                                 'min_diff_m': min_diff_m,
                                                 'max_diff_m': max_diff_m,
                                                 'std_diff_m': std_diff_m})

    return dict_output


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
