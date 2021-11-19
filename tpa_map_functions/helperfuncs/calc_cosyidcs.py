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

    a = np.where(s_actual_m >= coordinates_sxy_m[-1, 0])[0]

    if np.any(a):
        idx_min = int(np.min(a) - 1)

        print('WARNING: fix in calc_cosyidcs.py was triggered! '
              + 's_max is: {}; requested s-coordinate(s): {}'.format(coordinates_sxy_m[-1, 0], s_actual_m[a]))

        s_actual_m[a] = s_actual_m[idx_min]

    sectionid_change = np.concatenate((np.asarray([True]), np.diff(sectionid) != 0))
    coordinates_sxy_m = coordinates_sxy_m[sectionid_change]

    i = np.searchsorted(coordinates_sxy_m[:, 0], s_actual_m, side='right') - 1
    i[i < 0] = 0

    k = np.hstack((np.where(sectionid_change)[0][i][:, np.newaxis],
                   np.where(sectionid_change)[0][i + 1][:, np.newaxis]))

    return sectionid[sectionid_change][i], coordinates_sxy_m[i, 0], k


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    sectionid = np.asarray([1, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7])

    coordinates_sxy_m = np.hstack((np.arange(11)[:, np.newaxis], np.zeros((11, 2))))
    s_actual_m = np.linspace(1, 15.0, 11)

    a, b, c = calc_cosyidcs(sectionid=sectionid,
                            coordinates_sxy_m=coordinates_sxy_m,
                            s_actual_m=s_actual_m)

    pass
