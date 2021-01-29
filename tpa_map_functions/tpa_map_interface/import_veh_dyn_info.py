import numpy as np

"""
Created by: Leonhard Hermansdorfer
Created on: 05.09.2020
"""


def import_veh_dyn_info(filepath2localgg: str = "") -> np.ndarray:
    """
    .. description::
    This function imports the local acceleration limits specified by a 'localgg' file and checks validity of the
    imported data. The file contains the s-, x- and y-coordinates of the underlying reference line and the
    corresponding acceleration limits in longitudinal and lateral direction. The file consists of a single row,
    which results in a constant acceleration limit for the whole racetrack, or of multiple rows, which results in
    location-dependent accelerations limits.
    The file format is [s_m, x_m, y_m, ax_max_mps2, ay_max_mps2] with units [m, m, m, m/s^2, m/s^2].

    .. inputs::
    :param filepath2localgg:    absolute path to 'localgg' file which contains vehicle acceleration limits
    :type filepath2localgg:     str

    .. outputs::
    :return tpamap:             tire performance assessment (tpa) map containing the reference line and long./lat.
                                local acceleration limits
    :rtype tpamap:              np.ndarray
    """

    # raise error if no path is provided
    if not filepath2localgg:
        raise ValueError('Missing path to file which contains vehicle acceleration limits!')

    # load localgg data from csv file
    with open(filepath2localgg, 'rb') as fh:
        first_line = fh.readline()
        data_localggfile = np.loadtxt(fh, comments='#', delimiter=',')

    velocity_steps = []

    # read velocity steps from header
    try:
        for ele in str(first_line).split(','):
            if 'mps' in ele:
                velocity_steps.append(float(ele.split('__')[1].split('mps')[0]))

        count_veldep_columns = len(velocity_steps)

        if count_veldep_columns == 2:
            velocity_steps = []

    # catch error, if no velocity data is available
    except IndexError:
        count_veldep_columns = 2
        velocity_steps = []
        pass

    # Check Imported Data for Validity ---------------------------------------------------------------------------------

    # check dimension of localgg data
    if data_localggfile.ndim == 1:

        if data_localggfile.size < 5:
            raise ValueError('TPA MapInterface: wrong shape of localgg file data -> at least five columns required!')

        if data_localggfile.size != 3 + count_veldep_columns:
            raise ValueError('TPA MapInterface: wrong shape of localgg file data -> number of data columns and header entries does not match!')

#        if data_localggfile.size > 5:
#            print('WARNING: TPA MapInterface: shape of localgg file data -> more than five columns provided!')

        tpamap = np.hstack((np.zeros(3), data_localggfile[3:]))[np.newaxis, :]

    elif data_localggfile.ndim == 2:

        if data_localggfile.shape[1] < 5:
            raise ValueError('TPA MapInterface: wrong shape of localgg file data -> at least five columns required!')

        if data_localggfile.shape[1] != 3 + count_veldep_columns:
            raise ValueError('TPA MapInterface: wrong shape of localgg file data -> number of data columns and header entries does not match!')

#        if data_localggfile.shape[1] > 5:
#            print('WARNING: TPA MapInterface: shape of localgg file data -> more than five columns provided!')

        tpamap = data_localggfile

        # check validity of sxy-coordinates
        if np.any(tpamap[:, 0] < 0.0):
            raise ValueError('TPA MapInterface: one or more s-coordinate values are smaller than zero!')

        if np.any(np.diff(tpamap[:, 0]) <= 0.0):
            raise ValueError('TPA MapInterface: s-coordinates are not strictly monotone increasing!')

        # check whether endpoint and start point of s is close together in xy
        if not np.isclose(np.hypot(tpamap[0, 1] - tpamap[-1, 1], tpamap[0, 2] - tpamap[-1, 2]), 0.0):
            raise ValueError('TPA MapInterface: s-coordinates representing the race track are not closed; '
                             'first and last point are not equal!')

    else:
        raise ValueError("Localgg data file must provide one or two dimensions!")

    # check velocity data for validity
    if count_veldep_columns != 0:

        if velocity_steps[0::2] != velocity_steps[1::2]:
            raise ValueError('TPA MapInterface: data import: acceleration limits are not pairwise equal for same velocity!')

        if not np.all(np.diff(np.asarray(velocity_steps[0::2])) > 0):
            raise ValueError('TPA MapInterface: data import: velocity steps are not increasing monotonously!')

    # check local acceleration limits for validity
    if np.any(tpamap[:, 3:] > 40.0):
        raise ValueError('TPA MapInterface: max. acceleration limit in localgg file exceeds 40 m/s^2!')

    if np.any(tpamap[:, 3:] < 1.0):
        raise ValueError('TPA MapInterface: min. acceleration limit in localgg file is below 1 m/s^2!')

    return tpamap, velocity_steps[::2]


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
