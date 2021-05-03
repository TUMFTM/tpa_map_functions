import numpy as np

"""
Created by: Leonhard Hermansdorfer
Created on: 05.09.2020
"""


def calc_cosyidcs(sectionid: np.array,
                  coordinates_sxy_m: np.array,
                  s_actual_m: np.array) -> tuple:
    """Calculates section and section indices of current position (as s-coordinate).

    :param sectionid:                   section id array of race track
    :type sectionid: np.array
    :param coordinates_sxy_m:           s-coordinates of race track (from global raceline)
    :type coordinates_sxy_m: np.array
    :param s_actual_m:                  currently driven s-coordinate (from current trajectory)
    :type s_actual_m: np.array
    :return:                            section id of current position, starting s-coodinate of current section,
                                        start/end index of current section
    :rtype: tuple
    """

    sectionid_change = np.concatenate((np.asarray([True]), np.diff(sectionid) > 0))
    coordinates_sxy_m = coordinates_sxy_m[sectionid_change]

    i = np.searchsorted(coordinates_sxy_m[:, 0], s_actual_m, side='right') - 1
    i[i < 0] = 0

    k = np.hstack((np.where(sectionid_change)[0][i][:, np.newaxis],
                   np.where(sectionid_change)[0][i + 1][:, np.newaxis]))

    return sectionid[sectionid_change][i], coordinates_sxy_m[i, 0], k
