import numpy as np
import trajectory_planning_helpers as tph

"""
Created by: Leonhard Hermansdorfer
Created on: 05.09.2020
"""


def transform_coordinates_xy2s(coordinates_sxy_m: np.array,
                               position_m: np.array,
                               s_tot_m: float) -> np.array:

    distance_pathpoints = np.sqrt(np.diff(position_m[:, 0], 1) ** 2 + np.diff(position_m[:, 1], 1) ** 2)

    s_actual_m = np.zeros(position_m.shape[0])

    # tic_f = time.time()

    # match first entry of ego position on race line s-coordinate
    s_actual_m[0], _ = tph.path_matching_global.path_matching_global(path_cl=coordinates_sxy_m,
                                                                     ego_position=position_m[0, :])

    # TODO consider adding s_expected=self.lastcoordinate_s_m, s_range=40 (didn't increase performance)
    #  -> test again
    # self.lastcoordinate_s_m = s_actual_m[0]

    # sum up distance
    s_actual_m[1:] = s_actual_m[0] + np.cumsum(distance_pathpoints)

    # TODO write without for loop (use < on array + matrix multiplication)
    # after timing, this seems to be faster than version below
    for index, row in enumerate(s_actual_m):
        if row >= s_tot_m:
            s_actual_m[index] -= s_tot_m * np.floor(row / s_tot_m)
        elif row < 0:
            s_actual_m[index] += s_tot_m

    return s_actual_m
