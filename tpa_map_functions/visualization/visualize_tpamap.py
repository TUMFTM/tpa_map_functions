import numpy as np
import os.path
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

# import tikzplotlib

"""
Created by: Leonhard Hermansdorfer
Created on: 15.11.2019
"""


def plot_tpamap_as2dgrid(refline: np.array,
                         width_right: np.array,
                         width_left: np.array,
                         normvec_normalized: np.array,
                         filepath2tpamap: str = str(),
                         tpamap: np.array = None,
                         distance_scoord_labels: float = 200.0):
    """Loads and plots the acceleration limits of the tpa map into a 2d race track map.

    Loads tpamap csv-file which contains resampled reference line and corresponding acceleration limits.
    Loads traj_ltpl_cl csv-file which contains reference line and corresponding trackwidth information.

    Input
    :param path_dict: contains all paths to relevant folders and file within this software module
    :type path_dict: dict
    :param refline: reference line of the ltpl csv-file
    :type refline: np.array
    :param width_right: trackwidth to rigth of reference line
    :type width_right: np.array
    :param width_left: trackwidth to left of reference line
    :type width_left: np.array
    :param normvec_normalized: normal vectors to each coordinate of reference line (necessary to calculate trackboundary
                                coordinates)
    :type normvec_normalized: np.array

    Error
    :raises ValueError: [description]
    :raises ValueError: [description]
    :raises an: [description]
    :raises ValueError: [description]
    :raises error: [description]
    :raises ValueError: [description]

    Output
    ---
    """

    # raise error, if both filepath and reference line array is provided
    if bool(filepath2tpamap) and tpamap is not None:
        print('ERROR: path to tpamap file AND tpamap array provided! Only provide one input.')
        raise ValueError('path to tpamap file AND tpamap array provided!')

    # raise error, if neither filepath nor reference line array is provided
    elif not filepath2tpamap and tpamap is None:
        print('ERROR: neither path to tpamap file nor tpamap array provided! At least provide one input.')
        raise ValueError('neither path to tpamap file nor tpamap array provided!')

    # load tpamap data from file
    if bool(filepath2tpamap):
        with open(filepath2tpamap, 'r') as fh:
            tpamap = np.genfromtxt(fh, delimiter=',')

    # ------------------------------------------------------------------------------------------------------------------

    # calculate cumulative length of reference line
    s_refline_m = np.cumsum(np.sqrt(np.sum((np.square(np.diff(np.vstack((refline, refline[0]))[:, 0])),
                                            np.square(np.diff(np.vstack((refline, refline[0]))[:, 1]))), axis=0)))

    refline_concat = np.hstack((np.vstack((np.zeros(1), s_refline_m[:, np.newaxis])),
                                np.vstack((refline, refline[0]))))

    refline_closed = np.vstack((refline, refline[0:3, :]))

    # calculate xy-coordinates of left/right track boundaries
    trackboundary_right_m = refline + np.multiply(normvec_normalized, width_right[:, np.newaxis])
    trackboundary_left_m = refline - np.multiply(normvec_normalized, width_left[:, np.newaxis])
    trackboundary_right_m__closed = np.vstack((trackboundary_right_m, trackboundary_right_m[0:3]))
    trackboundary_left_m__closed = np.vstack((trackboundary_left_m, trackboundary_left_m[0:3]))

    # last value equals first entry, therefore discard
    ax_comb_mps2 = tpamap[:-1, -2]

    list_points2 = list()
    test_refline = list()
    tb_right_interp_m = list()
    tb_left_interp_m = list()

    bool_isdeleted_lastentry = False
    idx_min = 0

    for row in tpamap:

        idx_min_last = idx_min

        idx_min = np.argmin(np.sqrt(np.sum((
            np.square(row[1] - refline_concat[:, 1]),
            np.square(row[2] - refline_concat[:, 2])), axis=0)))

        if idx_min_last > 0 and idx_min == 0:
            idx_min = len(refline_concat) - 1

        if not(idx_min == idx_min_last):

            if bool_isdeleted_lastentry:
                i_add_start = 0
            else:
                i_add_start = 1

            for i_count in range(idx_min_last + i_add_start, idx_min + 1):
                test_refline.append(refline_concat[i_count, 1:])
                tb_right_interp_m.append(trackboundary_right_m__closed[i_count])
                tb_left_interp_m.append(trackboundary_left_m__closed[i_count])

        bool_isdeleted_lastentry = False

        # interpolate reference line and track boundaries
        if not(refline_concat[idx_min, 0] == row[0]):

            # calculate distance to coordinate before and after actual nearest coordinate
            # to determine between which coordinates to interploate
            dist_bw = np.sqrt(np.sum((np.square(row[1] - refline_closed[idx_min - 1, 0]),
                                      np.square(row[2] - refline_closed[idx_min - 1, 1]))))
            dist_fw = np.sqrt(np.sum((np.square(row[1] - refline_closed[idx_min + 1, 0]),
                                      np.square(row[2] - refline_closed[idx_min + 1, 1]))))

            if dist_bw > dist_fw:
                idx_add_start = 0
                idx_add_end = 2

            elif dist_bw < dist_fw:
                idx_add_start = -1
                idx_add_end = 1

                del test_refline[-1]
                del tb_right_interp_m[-1]
                del tb_left_interp_m[-1]

                bool_isdeleted_lastentry = True

            # raise an error when location of coordinate can not be identified with this method
            # TODO: avoid siatuation which would lead to error
            else:
                raise ValueError()

            dist = np.sqrt(np.sum((np.square(row[1] - refline_closed[idx_min + idx_add_start, 0]),
                                   np.square(row[2] - refline_closed[idx_min + idx_add_start, 1]))))

            x_vals = np.hstack((np.zeros(1), np.sqrt(np.sum((
                np.square(refline_closed[idx_min + idx_add_end - 1, 0]
                          - refline_closed[idx_min + idx_add_start, 0]),
                np.square(refline_closed[idx_min + idx_add_end - 1, 1]
                          - refline_closed[idx_min + idx_add_start, 1]))))))

            test_refline.append(np.hstack((
                np.interp(dist, x_vals, refline_closed[idx_min + idx_add_start:idx_min + idx_add_end, 0]),
                np.interp(dist, x_vals, refline_closed[idx_min + idx_add_start:idx_min + idx_add_end, 1]))))

            tb_right_interp_m.append(np.hstack((
                np.interp(dist, x_vals,
                          trackboundary_right_m__closed[idx_min + idx_add_start:idx_min + idx_add_end, 0]),
                np.interp(dist, x_vals,
                          trackboundary_right_m__closed[idx_min + idx_add_start:idx_min + idx_add_end, 1]))))

            tb_left_interp_m.append(np.hstack((
                np.interp(dist, x_vals,
                          trackboundary_left_m__closed[idx_min + idx_add_start:idx_min + idx_add_end, 0]),
                np.interp(dist, x_vals,
                          trackboundary_left_m__closed[idx_min + idx_add_start:idx_min + idx_add_end, 1]))))

        else:

            test_refline.append(refline[idx_min, :])
            tb_right_interp_m.append(trackboundary_right_m__closed[idx_min, :])
            tb_left_interp_m.append(trackboundary_left_m__closed[idx_min, :])

        list_points2.append(len(tb_left_interp_m) - 1)

    # check whether last entry is identical to first entry; if not, values of first entry are copied to last
    dist = np.sqrt(np.sum((np.square(test_refline[-1][0] - test_refline[0][0]),
                           np.square(test_refline[-1][1] - test_refline[0][1]))))

    if dist < 0.01:
        if np.sqrt(np.sum((np.square(test_refline[-2][0] - test_refline[-1][0]),
                           np.square(test_refline[-2][1] - test_refline[-1][1])))) < 0.01:
            del test_refline[-1]
            del tb_right_interp_m[-1]
            del tb_left_interp_m[-1]

        test_refline[-1] = test_refline[0]
        tb_right_interp_m[-1] = tb_right_interp_m[0]
        tb_left_interp_m[-1] = tb_left_interp_m[0]

    # transform lists to numpy arrays
    test_refline = np.array(test_refline)
    tb_right_interp_m = np.array(tb_right_interp_m)
    tb_left_interp_m = np.array(tb_left_interp_m)

    # testing ----------------------------------------------------------------------------------------------------------
    # calculate distance between interpolated reference line coordinates
    # to check whether the coordinates are equally spaced

    s_test = np.cumsum(np.sqrt(np.sum((np.square(np.diff(np.vstack((test_refline, test_refline[0]))[:, 0])),
                                       np.square(np.diff(np.vstack((test_refline, test_refline[0]))[:, 1]))), axis=0)))

    s_test = np.vstack((np.zeros(1), s_test[:, np.newaxis]))

    list_testing = []

    for i_counter in range(len(list_points2) - 1):
        list_testing.append(s_test[(list_points2[i_counter + 1])] - s_test[list_points2[i_counter]])

    list_testing = np.array(list_testing)

    # testing end ------------------------------------------------------------------------------------------------------

    # create list containing each polygon to plot;
    # polygons consist of the points of the respective left and right track boundary sections

    patches = []

    for i_row in range(len(list_points2) - 1):

        points = np.vstack((
            tb_right_interp_m[list_points2[i_row]:(list_points2[i_row + 1] + 1), :],
            np.flipud(tb_left_interp_m[list_points2[i_row]:(list_points2[i_row + 1] + 1), :])))

        polygon = Polygon(points, closed=True)
        patches.append(polygon)

    # raise error when number of polygons to plot and corresponding data for coloring the polygons are not equal
    if not(len(patches) == len(tpamap) - 1):
        raise ValueError('tpamap visualization - data mismatch')

    # plot figure ------------------------------------------------------------------------------------------------------
    fig_mainplot, ax = plt.subplots()

    # plot s-coordinate labels
    plotting_distance_m = np.arange(0, refline_concat[-1, 0], distance_scoord_labels)

    for int_counter, ele in enumerate(plotting_distance_m):
        array = np.asarray(refline_concat[:, 0])
        idx = (np.abs(array - ele)).argmin()

        plt.plot(refline_concat[idx, 1], refline_concat[idx, 2], 'bx')
        plt.annotate('s=' + str(plotting_distance_m.tolist()[int_counter]) + ' m',
                     (refline_concat[idx, 1], refline_concat[idx, 2]),
                     xytext=(0, 30), textcoords='offset points', ha='center', va='bottom', color='blue',
                     bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', color='b'))

    # plot areas with colors
    collection = PatchCollection(patches, cmap=plt.set_cmap('viridis'))
    collection.set_array(ax_comb_mps2)
    ax.add_collection(collection)

    cbar = fig_mainplot.colorbar(collection, ax=ax)
    cbar.set_ticks(np.arange(0, np.ceil(np.max(ax_comb_mps2)) * 1.01, 0.5).round(2).tolist())
    b_list = []

    for ele in np.arange(0, np.ceil(np.max(ax_comb_mps2)) * 1.01, 0.5).round(2).tolist():
        b_list.append(str(ele))

    cbar.set_ticklabels(b_list)
    cbar.set_label('vehicle acceleration limit in m/s^2')

    # plot track boundaries
    plt.plot(trackboundary_right_m[:, 0], trackboundary_right_m[:, 1], 'k')
    plt.plot(trackboundary_left_m[:, 0], trackboundary_left_m[:, 1], 'k')

    # plot reference line
    plt.plot(tpamap[:, 1], tpamap[:, 2], 'k', linestyle='None', marker='x', label='reference line - interp')
    plt.plot(refline[:, 0], refline[:, 1], 'r', label='reference line')

    plt.title('tire performance map')
    plt.legend()

    plt.xlabel('x in meters')
    plt.ylabel('y in meters')
    plt.axis('equal')


    # tikz specific settings
    # plt.draw()
    # fig_mainplot.canvas.draw()
    # fig_mainplot.canvas.flush_events()
    # tikzplotlib.save('tpa.tex')

    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    import sys

    # import custom modules
    path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0],
                            'tpa_map_functions')

    sys.path.append(path2tmf)

    import tpa_map_functions

    filepath2ltpl_refline = os.path.join(path2tmf, 'example_files', 'traj_ltpl_cl', 'traj_ltpl_cl_berlin.csv')
    filepath2tpamap = os.path.join(path2tmf, 'example_files', 'veh_dyn_info', 'localgg_varloc_constvel.csv')

    dict_output = tpa_map_functions.helperfuncs.preprocess_ltplrefline.\
        preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline)

    plot_tpamap_as2dgrid(filepath2tpamap=filepath2tpamap,
                         refline=dict_output['refline'],
                         width_right=dict_output['width_right'],
                         width_left=dict_output['width_left'],
                         normvec_normalized=dict_output['normvec_normalized'])
