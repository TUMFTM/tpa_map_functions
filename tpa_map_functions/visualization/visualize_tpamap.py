import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider

# import tikzplotlib

# import custom modules
path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions.helperfuncs.import_vehdyninfo

"""
Created by: Leonhard Hermansdorfer
Created on: 15.11.2019
"""


def visualize_tpamap(refline: np.array,
                     width_right: np.array,
                     width_left: np.array,
                     normvec_normalized: np.array,
                     filepath2tpamap: str = str(),
                     tpamap: np.array = None,
                     distance_scoord_labels: float = 400.0,
                     fig_handle=None):
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
        tpamap, vel_steps = tpa_map_functions.helperfuncs.import_vehdyninfo.\
            import_vehdyninfo(filepath2localgg=filepath2tpamap)

        section_id = tpamap[:, 0]  # noqa F841
        tpamap = tpamap[:, 1:]

    else:
        vel_steps = []

    # ------------------------------------------------------------------------------------------------------------------

    if refline.shape[1] != 3:

        # calculate cumulative length of reference line
        s_refline_m = np.cumsum(np.sqrt(np.sum((np.square(np.diff(np.vstack((refline, refline[0]))[:, 0])),
                                                np.square(np.diff(np.vstack((refline, refline[0]))[:, 1]))), axis=0)))

        refline_concat = np.hstack((np.vstack((np.zeros(1), s_refline_m[:, np.newaxis])),
                                    np.vstack((refline, refline[0]))))

    else:
        refline_concat = refline.copy()
        refline = refline[:, 1:3]

    refline_closed = np.vstack((refline, refline[0:3, :]))

    # calculate xy-coordinates of left/right track boundaries
    trackboundary_right_m = refline + np.multiply(normvec_normalized, width_right[:, np.newaxis])
    trackboundary_left_m = refline - np.multiply(normvec_normalized, width_left[:, np.newaxis])
    trackboundary_right_m__closed = np.vstack((trackboundary_right_m, trackboundary_right_m[0:3]))
    trackboundary_left_m__closed = np.vstack((trackboundary_left_m, trackboundary_left_m[0:3]))

    list_points2 = list()
    test_refline = list()
    tb_right_interp_m = list()
    tb_left_interp_m = list()

    bool_isdeleted_lastentry = False
    idx_min = 0

    # match tpamap coordinates onto original reference line for plotting
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

    # define plot functions --------------------------------------------------------------------------------------------

    def plot_ax1(acomb_mps2: np.array,
                 bool_is_firstcall: bool = False,
                 xlim: list() = [],
                 ylim: list() = []):

        # plot track boundaries
        ax1.plot(trackboundary_right_m[:, 0], trackboundary_right_m[:, 1], 'k')
        ax1.plot(trackboundary_left_m[:, 0], trackboundary_left_m[:, 1], 'k')

        # plot reference line
        # ax1.plot(tpamap[:, 1], tpamap[:, 2], 'k', linestyle='None', marker=',', label='reference line - interp')
        ax1.plot(refline[:, 0], refline[:, 1], 'r', label='reference line')

        # plot s-coordinate labels
        plotting_distance_m = np.arange(0, refline_concat[-1, 0], distance_scoord_labels)

        for int_counter, ele in enumerate(plotting_distance_m):
            idx = (np.abs(refline_concat[:, 0] - ele)).argmin()

            ax1.plot(refline_concat[idx, 1], refline_concat[idx, 2], 'bo')
            ax1.annotate('s=' + str(plotting_distance_m.tolist()[int_counter]) + ' m',
                         (refline_concat[idx, 1], refline_concat[idx, 2]),
                         xytext=(0, 30), textcoords='offset points', ha='center', va='bottom', color='blue',
                         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', color='b'))

        # plot areas with colors
        collection = PatchCollection(patches, cmap=plt.set_cmap('viridis'))
        collection.set_array(acomb_mps2)
        ax1.add_collection(collection)
        collection.set_clim(y_limits)

        list_cblabels = np.arange(0, y_limits[1] * 1.01, 1.0).round(2).tolist()

        if bool_is_firstcall:
            cbar = plt.colorbar(collection, ax=ax1)

            cbar.set_ticks(list_cblabels)
            b_list = []

            for ele in list_cblabels:
                b_list.append(str(ele))

            cbar.set_ticklabels(b_list)
            cbar.set_label('vehicle acc. limit in m/s^2')

        ax1.legend()
        ax1.set_title('tire performance map')
        ax1.set_xlabel('x in meters')
        ax1.set_ylabel('y in meters')
        ax1.axis('equal')

        if xlim and ylim:
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)

    # -----------------------------------------------------

    def plot_ax2(no_vel: list):

        ax2.step(tpamap[:, 0], tpamap[:, 3 + (no_vel - 1) * 2], where='post', label='long. acc.')
        ax2.step(tpamap[:, 0], tpamap[:, 4 + (no_vel - 1) * 2], where='post', label='lat. acc.')

        ax2.grid()
        ax2.legend()
        ax2.set_xlabel('track position in meters')
        ax2.set_ylabel('long./lat. acc. in m/s^2')
        ax2.set_xlim((tpamap[0, 0], tpamap[-1, 0]))
        ax2.set_ylim((y_limits))

    # plot figure ------------------------------------------------------------------------------------------------------
    bool_add_subplot = True
    bool_add_slider = True

    if len(vel_steps) == 0:
        bool_add_slider = False

    # last value equals first entry, therefore discard
    acomb_mps2 = np.divide(np.sum(tpamap[:-1, 3:5], axis=1), 2)

    # set y limits of colobar denpending on data available
    if bool_add_subplot and bool_add_slider:
        y_limits = [max(np.min(tpamap[:, 3:]) - 2, 0), np.max(tpamap[:, 3:]) + 2]
    else:
        y_limits = [max(np.min(tpamap[:, 3:5]) - 2, 0), np.max(tpamap[:, 3:5]) + 2]

    # use existing figure when provided (e.g. when GUI is running)
    if fig_handle is None:
        fig = plt.figure(figsize=(14.5, 8))
    else:
        fig = fig_handle

    # create subplots if activated
    if bool_add_subplot:
        ax1 = plt.subplot(3, 1, (1, 2))
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.90, top=0.9, wspace=None, hspace=0.3)
    else:
        ax1 = plt.subplot()

    plot_ax1(acomb_mps2=acomb_mps2, bool_is_firstcall=True)

    # tikz specific settings
    # plt.draw()
    # fig_mainplot.canvas.draw()
    # fig_mainplot.canvas.flush_events()
    # tikzplotlib.save('tpa.tex')

    if bool_add_subplot:
        ax2 = plt.subplot(3, 1, 3)

        plot_ax2(no_vel=1)

        if bool_add_slider:

            plt.subplots_adjust(bottom=0.15)
            axcolor = 'lightgoldenrodyellow'
            axvel = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
            svel = Slider(axvel, 'velocity_step', valmin=1, valmax=len(vel_steps), valinit=1, valstep=1)

            def update_plot(val):
                no_vel = svel.val
                print("plot data for " + str(vel_steps[no_vel - 1]) + " mps")

                ax1_xlim = ax1.get_xlim()
                ax1_ylim = ax1.get_ylim()

                ax1.clear()
                acomb_mps2 = np.divide(np.sum(tpamap[:-1, (3 + (no_vel - 1) * 2):(5 + (no_vel - 1) * 2)], axis=1), 2)
                plot_ax1(acomb_mps2=acomb_mps2, xlim=list(ax1_xlim), ylim=list(ax1_ylim))

                ax2.clear()
                plot_ax2(no_vel=no_vel)

                fig.canvas.draw_idle()

            svel.on_changed(update_plot)

    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
