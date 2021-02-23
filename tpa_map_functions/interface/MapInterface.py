import numpy as np
import zmq
import time
import os.path
import sys
import ad_interface_functions
import trajectory_planning_helpers as tph

path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions.interface.import_vehdyninfo

"""
Created by: Leonhard Hermansdorfer
Created on: 10.12.2019
"""


class MapInterface:
    """Provides an interface to the local trajectory planner to access local acceleration limitations.

    The MapInterface class provides an interface between the local trajectory planning module and the tire performance
    assessment module to fetch and distribute local acceleration limitations used for trajectory planning.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Constructor ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 filepath2localgg: str,
                 zmq_opts_sub_tpa: dict = None,
                 bool_enable_interface2tpa: bool = False,
                 bool_enable_interpolation: bool = False,
                 bool_enable_velocitydependence: bool = False):
        """Initializes MapInterface class to provide an interface for accessing the local acceleration limits and to
           enable communication to tpa module for online update of acc. limits.

        Input
        :param filepath2localgg: filepath to csv file where local acceleration limits are stored
        :type filepath2localgg: str
        :param zmq_opts_sub_tpa: zmq options for interface to tire performance assessment (tpa) module, defaults to None
        :type zmq_opts_sub_tpa: dict, optional
        :param bool_enable_interface2tpa: enables/disables (True/False) communication to tpa module, defaults to False
        :type bool_enable_interface2tpa: bool, optional
        :param bool_enable_interpolation: enables/disables (True/False) interpolation of acceleration limits for the
                                          actual s-position with regards to the s-coordinates of the tpa map,
                                          defaults to False
        :type bool_enable_interpolation: bool, optional
        :param bool_enable_velocitydependence: enables/disables (True/False) velocity dependent acceleration limits,
                                               defaults to False
        :type bool_enable_velocitydependence: bool, optional

        Error
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises FileExistsError: [description]
        """

        # write input to class instance
        self.__zmq_opts_sub_tpa = zmq_opts_sub_tpa
        self.__bool_enable_interface2tpa = bool_enable_interface2tpa
        self.__bool_enable_interpolation = bool_enable_interpolation
        self.__bool_enable_velocitydependence = bool_enable_velocitydependence

        # flag to indicate whether or not a tpa map was received via zmq
        self.__bool_received_tpamap = False

        self.s_tot_m = 0.0
        self.coordinates_sxy_m = np.zeros((1, 3))
        self.coordinates_sxy_m_extended = np.zeros((1, 3))
        self.localgg_mps2 = np.zeros((1, 2))

        # contains latest received local acceleration limits
        self.__localgg_lastupdate = np.zeros((1, 2))

        # TODO integrate into global s matching
        self.lastcoordinate_s_m = None

        self.__s_egopos_m = None
        self.__s_ltpllookahead_m = None

        # safety distances which is added/subtracted to/from current trajectory begin/ending
        # add to distance of current planning horizon
        self.__s_lookahead_safety_m = 50.0
        # subtract from current vehicle position
        self.__s_lookback_safety_m = 50.0

        # --------------------------------------------------------------------------------------------------------------
        # Read Data File Containing Tire Performance Assessment Map ----------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        tpamap, velocity_steps = tpa_map_functions.interface.import_vehdyninfo.import_vehdyninfo(
            filepath2localgg=filepath2localgg)

        self.coordinates_sxy_m = tpamap[:, 0:3]

        # set data mode to global variable or global constant for further processing
        if tpamap.shape[0] > 1:
            self.data_mode = 'global_variable'
        else:
            self.data_mode = 'global_constant'

        # Check Localgg Data with Velocity Dependence ------------------------------------------------------------------

        if not bool(velocity_steps) and self.__bool_enable_velocitydependence:
            raise ValueError('TPA MapInterface: velocity dependence is enabled, but no velocity dependent acc. limits '
                             'are provided in inputs file!')

        elif bool(velocity_steps) and not self.__bool_enable_velocitydependence:
            raise ValueError('TPA MapInterface: velocity dependent acc. limits are provided in file, but velocity '
                             'dependence is disbaled!')

        if not bool(velocity_steps):
            self.__count_velocity_steps = 1

        else:
            self.__count_velocity_steps = int(len(velocity_steps))

        velocity_steps.insert(0, 0.0)
        self.velocity_steps = np.asarray(velocity_steps)

        # if true, add all local acc. limits including velocity dependence
        if self.__bool_enable_velocitydependence:
            self.localgg_mps2 = tpamap[:, 3:]

        else:
            self.localgg_mps2 = tpamap[:, 3:5]

        # --------------------------------------------------------------------------------------------------------------
        # Initialize Communication: ZMQ --------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if self.__zmq_opts_sub_tpa and self.__bool_enable_interface2tpa:
            self.zmq_context = zmq.Context()

            # Init ZMQ sockets for communication with tire performance assessment module
            self.sock_zmq_rcv_tpa = self.zmq_context.socket(zmq.SUB)

            self.sock_zmq_rcv_tpa.connect("tcp://%s:%s" % (self.__zmq_opts_sub_tpa["ip"],
                                                           self.__zmq_opts_sub_tpa["port_data"]))

            self.sock_zmq_rcv_tpa.setsockopt_string(zmq.SUBSCRIBE, self.__zmq_opts_sub_tpa["topic"])

            # wait a short time until all sockets are really bound (ZMQ specific problem)
            time.sleep(0.5)

        # skip when global constant values are used
        if self.data_mode == 'global_variable' and not self.__bool_enable_interface2tpa:
            self.format_rawtpamap()

            self.localgg_extended = np.vstack((self.localgg_mps2[-2, :], self.localgg_mps2))

        elif self.data_mode == 'global_variable' and self.__bool_enable_interface2tpa:
            raise FileExistsError('localgg file in inputs/veh_dyn_info does contain multiple entries. ',
                                  'This is not allowed when interface to tpa module is enabled. ',
                                  'Only put one row with initial values in file.')

    # ------------------------------------------------------------------------------------------------------------------
    # Destructor -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __del__(self):
        """Clears all network related stuff.
        """

        try:
            self.sock_zmq_rcv_tpa.close()
            self.zmq_context.term()

            time.sleep(0.5)
            print("TPA MapInterface: Sockets closed!")

        except AttributeError:
            print("TPA MapInterface: closed!")

    # ------------------------------------------------------------------------------------------------------------------
    # Custom Class Methods ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def format_rawtpamap(self):
        """Formats s-coordinate of raw tpamap data loaded from file or received via zmq from tpa module.
        """

        self.s_tot_m = self.coordinates_sxy_m[-1, 0]
        self.coordinates_sxy_m_extended = np.vstack((self.coordinates_sxy_m[-2, :], self.coordinates_sxy_m))
        self.coordinates_sxy_m_extended[0, 0] = self.coordinates_sxy_m[-2, 0] - self.coordinates_sxy_m[-1, 0]

    # ------------------------------------------------------------------------------------------------------------------

    def get_acclim_tpainterface(self,
                                position_m: np.array,
                                position_mode: str,
                                velocity_mps: np.array = np.asarray([])) -> np.array:
        """Provides an interface between the local acceleration limit map and the trajectory planer

        Input
        :param position_m: contains xy- or s-coordinates of planed path for request of local acceleration limits
        :type position_m: np.array
        :param position_mode: specifies whether xy-coordinates or s-coordinates are provided via 'position_m'
        :type position_mode: str
        :param velocity_mps: contains the vehicle velocity for which the local acceleration limits should get
                             calculated, defaults to np.asarray([])
        :type velocity_mps: np.array, optional

        Error
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises ValueError: [description]
        :raises error: [description]
        :raises ValueError: [description]

        Output
        :return localgg: contains longitudinal and lateral acceleration limit for every requested position
        :rtype: np.array
        """

        # --------------------------------------------------------------------------------------------------------------
        # Check function arguments for validity ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # check if position mode is valid
        if position_mode == 'xy-cosy':
            count_columns = 2

        elif position_mode == 's-cosy':
            count_columns = 1

        else:
            raise ValueError('TPA MAPInterface: unknown position mode during local acceleration limit request!')

        # check if number of columns is valid depending on what position information is provided
        if position_m.ndim == 1:
            count_rows = 1

            if position_m.size != count_columns:
                raise ValueError('TPA MapInterface: wrong shape of position data during local gg request!')

        elif position_m.ndim == 2:
            count_rows = position_m.shape[0]

            if position_m.shape[1] != count_columns:
                raise ValueError('TPA MapInterface: wrong shape of position data during local gg request!')

        # check if velocity dependence is enabled and velocity values are provided

        # TODO: uncomment when trajectory planner is ready
        # if not self.__bool_enable_velocitydependence and velocity_mps.size != 0:
        #    raise ValueError('TPA MapInterface: velocity for velocity dependent acc. limits request is provided, but '
        #                     'velocity dependence is disbaled!')

        if self.__bool_enable_velocitydependence and velocity_mps.size == 0:
            raise ValueError('TPA MapInterface: velocity dependence is enabled, but no velocity is provided for '
                             'request!')

        # TODO check if velocity array has correct shape and dimension

        # --------------------------------------------------------------------------------------------------------------
        # Fetch location-dependent and -independent acceleration limits ------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # calculate location-independent acceleration limits ('global constant') ---------------------------------------
        if self.data_mode == 'global_constant':

            if self.__bool_enable_velocitydependence:
                localgg = np.ones((count_rows, 2))

                ax = np.interp(velocity_mps, self.velocity_steps[1:], self.localgg_mps2[0][0::2])
                ay = np.interp(velocity_mps, self.velocity_steps[1:], self.localgg_mps2[0][1::2])

                localgg = np.hstack((ax, ay))

            else:
                localgg = np.ones((count_rows, 2)) * self.localgg_mps2[0]

        # calculate location-dependent acceleration limits ('global variable') -----------------------------------------
        elif self.data_mode == 'global_variable':

            # calculate s-coordinate when xy-coordinates are provided
            if position_mode == 'xy-cosy':

                distance_pathpoints = np.sqrt(np.diff(position_m[:, 0], 1) ** 2 + np.diff(position_m[:, 1], 1) ** 2)

                s_actual_m = np.zeros(position_m.shape[0])

                # tic_f = time.time()

                # match first entry of ego position on race line s-coordinate
                s_actual_m[0], _ = tph.path_matching_global.path_matching_global(path_cl=self.coordinates_sxy_m,
                                                                                 ego_position=position_m[0, :])

                # TODO consider adding s_expected=self.lastcoordinate_s_m, s_range=40 (didn't increase performance)
                #  -> test again
                # self.lastcoordinate_s_m = s_actual_m[0]

                # sum up distance
                s_actual_m[1:] = s_actual_m[0] + np.cumsum(distance_pathpoints)

                # TODO write without for loop (use < on array + matrix multiplication)
                # after timing, this seems to be faster than version below
                for index, row in enumerate(s_actual_m):
                    if row >= self.s_tot_m:
                        s_actual_m[index] -= self.s_tot_m
                    elif row < 0:
                        s_actual_m[index] += self.s_tot_m

                # s_actual_m = s_actual_m \
                #     - np.multiply((s_actual_m > self.s_tot_m), self.s_tot_m) \
                #     + np.multiply((s_actual_m < 0), self.s_tot_m)

            else:
                s_actual_m = np.hstack(position_m)

            # save s-coordinate of first (position of vehicle) and last (current planning horizon) entry of trajectory
            self.__s_egopos_m = s_actual_m[0]
            self.__s_ltpllookahead_m = s_actual_m[-1]

            # if True, interpolate acceleration limits of actual s-position between given s-coordinates of map
            if self.__bool_enable_interpolation:

                # initialize empty local gg array containing one column for each velocity step
                ax_out = np.zeros((s_actual_m.shape[0], self.__count_velocity_steps * 2))
                idx_ax_out = 0

                for row in s_actual_m:

                    idx = np.argmin(np.abs(self.coordinates_sxy_m[:, 0] - row)) + 1

                    if (self.coordinates_sxy_m_extended[idx, 0] - row) > 0:
                        idx -= 1
                        if idx < 0:
                            idx = self.coordinates_sxy_m.shape[0]

                    grad = np.divide((row - self.coordinates_sxy_m_extended[idx, 0]),
                                     (self.coordinates_sxy_m_extended[idx + 1, 0]
                                     - self.coordinates_sxy_m_extended[idx, 0]))

                    # Check Neighbouring Cells -------------------------------------------------------------------------
                    # check neighbouring cell values to always guarantee a conservative acceleration limit

                    # get information of neighbouring data separately for ax and ay
                    bool_idx_isequal_idxplus = self.localgg_extended[idx] == self.localgg_extended[idx + 1]
                    bool_idx_greater_idxplus = self.localgg_extended[idx] > self.localgg_extended[idx + 1]
                    bool_idx_smaller_idxplus = self.localgg_extended[idx] < self.localgg_extended[idx + 1]
                    bool_idxminus_isequal_idx = self.localgg_extended[idx - 1] == self.localgg_extended[idx]
                    bool_idxminus_smaller_idx = self.localgg_extended[idx - 1] < self.localgg_extended[idx]
                    bool_idxminus_greater_idx = self.localgg_extended[idx - 1] > self.localgg_extended[idx]

                    # handle all cases where current value is greater than next value
                    if np.any(bool_idx_greater_idxplus):

                        # handle case where last value is smaller than current value
                        if np.any(np.logical_and(bool_idx_greater_idxplus, bool_idxminus_smaller_idx)):
                            ax_out[idx_ax_out, np.logical_and(bool_idx_greater_idxplus, bool_idxminus_smaller_idx)] = \
                                (1 - grad) \
                                * self.localgg_extended[idx - 1,
                                                        np.logical_and(bool_idx_greater_idxplus,
                                                                       bool_idxminus_smaller_idx)] \
                                + grad * self.localgg_extended[idx + 1,
                                                               np.logical_and(bool_idx_greater_idxplus,
                                                                              bool_idxminus_smaller_idx)]

                        # handle case where last value is greater than current value
                        if np.any(np.logical_and(bool_idx_greater_idxplus, bool_idxminus_greater_idx)):
                            ax_out[idx_ax_out, np.logical_and(bool_idx_greater_idxplus, bool_idxminus_greater_idx)] = \
                                (1 - grad) \
                                * self.localgg_extended[idx,
                                                        np.logical_and(bool_idx_greater_idxplus,
                                                                       bool_idxminus_greater_idx)] \
                                + grad * self.localgg_extended[idx + 1,
                                                               np.logical_and(bool_idx_greater_idxplus,
                                                                              bool_idxminus_greater_idx)]

                        # TODO test!
                        # handle case where last value is equal to current value
                        if np.any(np.logical_and(bool_idx_greater_idxplus, bool_idxminus_isequal_idx)):
                            ax_out[idx_ax_out, np.logical_and(bool_idx_greater_idxplus, bool_idxminus_isequal_idx)] = \
                                (1 - grad) \
                                * self.localgg_extended[idx,
                                                        np.logical_and(bool_idx_greater_idxplus,
                                                                       bool_idxminus_isequal_idx)] \
                                + grad * self.localgg_extended[idx + 1,
                                                               np.logical_and(bool_idx_greater_idxplus,
                                                                              bool_idxminus_isequal_idx)]

                    # handle all cases where current value is smaller than next value
                    if np.any(bool_idx_smaller_idxplus):

                        # handle case where last value is smaller than current value
                        if np.any(np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_smaller_idx)):
                            ax_out[idx_ax_out, np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_smaller_idx)] = \
                                (1 - grad) \
                                * self.localgg_extended[idx - 1,
                                                        np.logical_and(bool_idx_smaller_idxplus,
                                                                       bool_idxminus_smaller_idx)] \
                                + grad * self.localgg_extended[idx,
                                                               np.logical_and(bool_idx_smaller_idxplus,
                                                                              bool_idxminus_smaller_idx)]

                        # handle case where last value is greater than current value
                        if np.any(np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_greater_idx)):
                            ax_out[idx_ax_out, np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_greater_idx)] = \
                                self.localgg_extended[idx, np.logical_and(bool_idx_smaller_idxplus,
                                                                          bool_idxminus_greater_idx)]

                        # TODO test!
                        # handle case where last value is equal to current value
                        if np.any(np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_isequal_idx)):

                            ax_out[idx_ax_out, np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_isequal_idx)] = \
                                self.localgg_extended[idx, np.logical_and(bool_idx_smaller_idxplus,
                                                                          bool_idxminus_isequal_idx)]

                    # handle all cases where current value is equal to next value
                    if np.any(bool_idx_isequal_idxplus):
                        ax_out[idx_ax_out, bool_idx_isequal_idxplus] = self.localgg_extended[idx,
                                                                                             bool_idx_isequal_idxplus]

                    idx_ax_out += 1

                # time_globalpath = time.time() - tic_f
                # logging.debug('time to get local acceleration limits: ' + str(time_globalpath))

            # if False, no interpolation is used; values of the corresponding tpamap section are taken
            else:
                idx_list = []

                for row in s_actual_m:
                    idx = np.argmin(np.abs(self.coordinates_sxy_m[:, 0] - row)) + 0

                    if (self.coordinates_sxy_m_extended[idx + 1, 0] - row) > 0:
                        idx -= 1

                        if idx < 0:
                            idx = self.coordinates_sxy_m.shape[0]

                    idx_list.append(idx)

                ax_out = self.localgg_mps2[idx_list, :]

            # if velocity dependence is enabled, the local acceleration limits are interpolated
            if self.__bool_enable_velocitydependence:

                ax = []
                ay = []

                for i, row in enumerate(ax_out):
                    ax.append(np.interp(velocity_mps[i], self.velocity_steps[1:], row[0::2]))
                    ay.append(np.interp(velocity_mps[i], self.velocity_steps[1:], row[1::2]))

                localgg = np.hstack((ax, ay))

            else:
                localgg = ax_out.copy()

        # raise error, if shape of return localgg is not equal to input trajectory; must have same length
        if position_m.shape[0] != localgg.shape[0]:
            raise ValueError('TPA MapInterface: number of rows of arrays for position request (input) and localgg '
                             '(output) do not match!')

        return localgg

    # ------------------------------------------------------------------------------------------------------------------

    def update(self):
        """Updates the MapInterface class with data from tire performance assessment module

        This function receives data from tpa module to update local acceleration limits. This function overwrites
        initial acceleration limits and handles the tpamap update which excludes the are within the trajectory planning
        horizon. This exclusion is necessary to avoid updating local acceleration limits leading to recursive
        infeasibility.
        """

        # jump update process when communication is disabled
        if self.__bool_enable_interface2tpa:

            # receive latest tire performance assessment data via ZMQ --------------------------------------------------
            data_tpainterface = ad_interface_functions.zmq_import.zmq_import(sock=self.sock_zmq_rcv_tpa, blocking=False)

            # check whether received tpa data is empty
            if data_tpainterface is not None:

                # check whether tpamap coordinates where already received
                if not self.__bool_received_tpamap:
                    self.coordinates_sxy_m = data_tpainterface[:, 0:3]

                    self.format_rawtpamap()

                    self.__bool_received_tpamap = True

                    # if current data mode is global_constant and data is received, switch to global_variable
                    if self.data_mode == 'global_constant':
                        self.localgg_mps2 = np.ones((self.coordinates_sxy_m.shape[0], 1)) * self.localgg_mps2
                        self.localgg_extended = np.vstack((self.localgg_mps2[-2, :], self.localgg_mps2))
                        self.data_mode = 'global_variable'

                self.__localgg_lastupdate = data_tpainterface[:, 3:5]

                # update stored tpamap besides planning horizon of local trajectory planer -----------------------------
                if self.__s_egopos_m is not None and self.__s_ltpllookahead_m is not None:

                    s_horizon_fw_m = self.__s_ltpllookahead_m + self.__s_lookahead_safety_m
                    s_horizon_bw_m = self.__s_egopos_m - self.__s_lookback_safety_m

                    if s_horizon_fw_m > self.s_tot_m:
                        s_horizon_fw_m -= self.s_tot_m

                    if s_horizon_bw_m < 0:
                        s_horizon_bw_m += self.s_tot_m

                    idx_start = np.argmin(np.abs(s_horizon_fw_m - self.coordinates_sxy_m[:, 0]))
                    idx_end = np.argmin(np.abs(s_horizon_bw_m - self.coordinates_sxy_m[:, 0]))

                    # TODO: abort when track is short that planning horizon "overtakes" ego position

                    if idx_start >= idx_end:
                        self.localgg_mps2[idx_start:, :] = self.__localgg_lastupdate[idx_start:, :].copy()
                        self.localgg_mps2[:idx_end, :] = self.__localgg_lastupdate[:idx_end, :].copy()

                    else:
                        self.localgg_mps2[idx_start:idx_end, :] = self.__localgg_lastupdate[idx_start:idx_end, :].copy()

                    self.localgg_extended = np.vstack((self.localgg_mps2[-2, :], self.localgg_mps2))

                    """
                    logging.debug("Current Tire Performance Parameters: "
                                 "max ax {}, min ax {}, max ay {}, min ay {}".format(np.max(localgg_mps2[:, 1]),
                                                                                     np.min(localgg_mps2[:, 1]),
                                                                                     np.max(localgg_mps2[:, 2]),
                                                                                     np.min(localgg_mps2[:, 2])))
                    """


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
