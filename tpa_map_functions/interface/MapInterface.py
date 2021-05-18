import numpy as np
import zmq
import time
import os.path
import sys
import ad_interface_functions

path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions.helperfuncs.import_vehdyninfo

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

        # temporary scale factor, which is updated via self.scale_acclimits_lapwise()
        self.__scalefactor_tmp = 1.0

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

        # variables for strategy module
        self.bool_isactivate_strategy = False
        self.bool_switchedoff_strat = False

        # --------------------------------------------------------------------------------------------------------------
        # Read Data File Containing Tire Performance Assessment Map ----------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        tpamap, velocity_steps = tpa_map_functions.helperfuncs.import_vehdyninfo.\
            import_vehdyninfo(filepath2localgg=filepath2localgg)

        self.section_id = tpamap[:, 0]

        # set data mode to global variable or global constant for further processing
        if tpamap.shape[0] > 1:
            self.data_mode = 'global_variable'
            self.sectionid_change = np.concatenate((np.asarray([True]), np.diff(self.section_id) > 0))
            self.coordinates_sxy_orignal_m = tpamap[:, 1:4]
            self.coordinates_sxy_m = self.coordinates_sxy_orignal_m[self.sectionid_change]

        else:
            self.data_mode = 'global_constant'
            self.sectionid_change = np.asarray([True])
            self.coordinates_sxy_orignal_m = tpamap[:, 1:4]
            self.coordinates_sxy_m = tpamap[:, 1:4]

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
            self.localgg_mps2 = tpamap[self.sectionid_change, 4:]

        else:
            self.localgg_mps2 = tpamap[self.sectionid_change, 4:6]

        # create separate localgg array for strategy updates
        self.localgg_strat_mps2 = self.localgg_mps2.copy()

        # skip when global constant values are used
        if self.data_mode == 'global_variable' and not self.__bool_enable_interface2tpa:
            self.format_rawtpamap()

        # check whether tpa-map file contains only one row (const. location)
        # and one value for each ax and ay (const. vel)
        if self.__bool_enable_interface2tpa and not (self.data_mode == 'global_constant'
                                                     and self.__count_velocity_steps == 1):
            raise ValueError('tpa-map file in inputs/veh_dyn_info has wrong shape!\n'
                             + 'When interface to tpa module is enabled, only provide one row of data and no '
                             + 'velocity-dependent acceleraion limits!\n'
                             + 'Shape of input data must be 1x6')

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

    # @profile
    def get_acclim_tpainterface(self,
                                position_m: np.array,
                                position_mode: str,
                                velocity_mps: np.array = np.asarray([])) -> np.array:
        """Provides an interface between the local acceleration limit map and the trajectory planer

        :param position_m: contains xy- or s-coordinates of planed path for request of local acceleration limits
        :type position_m: np.array
        :param position_mode: specifies whether xy-coordinates or s-coordinates are provided via 'position_m'
        :type position_mode: str
        :param velocity_mps: contains the vehicle velocity for which the local acceleration limits should get
                             calculated, defaults to np.asarray([])
        :type velocity_mps: np.array, optional

        :return localgg: contains longitudinal and lateral acceleration limit for every requested position
        :rtype: np.array
        """

        # --------------------------------------------------------------------------------------------------------------
        # Check function arguments for validity ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if position_mode != 'emergency':

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
        if not self.__bool_enable_velocitydependence and velocity_mps.size != 0:

            if not self.__bool_enable_interface2tpa:
                raise ValueError('TPA MapInterface: velocity for velocity dependent acc. limits request is provided, '
                                 + 'but velocity dependence is disbaled!')
            else:
                print('TPA MapInterface: WARNING: velocity for velocity dependent acc. limits request is provided, '
                      + 'but MapInterface has not received a velocity-dependent tpa-map!')

        if self.__bool_enable_velocitydependence and velocity_mps.size == 0:
            raise ValueError('TPA MapInterface: velocity dependence is enabled, but no velocity is provided for '
                             'request!')

        # TODO check if velocity array has correct shape and dimension

        # --------------------------------------------------------------------------------------------------------------
        # Handle request for emergency trajectory generation -----------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # use max. acc, limits for emergency trajectory calculation (not reduced limits from strategy)
        if position_mode == 'emergency':

            # use min. acc. limit value for each velocity step
            localgg_emergency_min = np.min(self.localgg_mps2[:, :], axis=0)

            if self.__bool_enable_velocitydependence:

                version = 2

                if version == 1:
                    ax = []
                    ay = []

                    ax_tmp = np.min(self.localgg_mps2[:, 0::2], axis=0)
                    ay_tmp = np.min(self.localgg_mps2[:, 1::2], axis=0)

                    for ele in velocity_mps:
                        ax.append(np.interp(ele, self.velocity_steps[1:], ax_tmp))
                        ay.append(np.interp(ele, self.velocity_steps[1:], ay_tmp))

                elif version == 2:

                    ax, ay = self.interp_velocitysteps(x=velocity_mps,
                                                       xp=np.hstack((self.velocity_steps, 150)),
                                                       fp=np.concatenate((localgg_emergency_min[:2],
                                                                          localgg_emergency_min,
                                                                          localgg_emergency_min[-2:])))

                return np.hstack((np.asarray(ax)[:, np.newaxis], np.asarray(ay)[:, np.newaxis]))

            else:
                return np.hstack((localgg_emergency_min[0], localgg_emergency_min[1]))[np.newaxis, :]

        # --------------------------------------------------------------------------------------------------------------
        # Fetch location-dependent and -independent acceleration limits ------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # use tpa-map with updates from strategy if active, else use original tpa-map
        if self.bool_isactivate_strategy:
            localgg_mps2 = self.localgg_strat_mps2

        else:
            localgg_mps2 = self.localgg_mps2 * self.__scalefactor_tmp

        # calculate location-independent acceleration limits ('global constant') ---------------------------------------
        if self.data_mode == 'global_constant':

            if self.__bool_enable_velocitydependence:
                localgg_out = np.ones((count_rows, 2))

                ax = np.interp(velocity_mps, self.velocity_steps[1:], localgg_mps2[0][0::2])
                ay = np.interp(velocity_mps, self.velocity_steps[1:], localgg_mps2[0][1::2])

                localgg_out = np.hstack((ax, ay))

            else:
                localgg_out = np.ones((count_rows, 2)) * localgg_mps2[0]

        # calculate location-dependent acceleration limits ('global variable') -----------------------------------------
        elif self.data_mode == 'global_variable':

            # calculate s-coordinate when xy-coordinates are provided
            if position_mode == 'xy-cosy':

                s_actual_m = tpa_map_functions.helperfuncs.transform_coordinates_xy2s.\
                    transform_coordinates_xy2s(coordinates_sxy_m=self.coordinates_sxy_orignal_m,
                                               position_m=position_m,
                                               s_tot_m=self.s_tot_m)

            else:
                s_actual_m = np.hstack(position_m)

            # save s-coordinate of first (position of vehicle) and last (current planning horizon) entry of trajectory
            self.__s_egopos_m = s_actual_m[0]
            self.__s_ltpllookahead_m = s_actual_m[-1]

            # if True, interpolate acceleration limits of actual s-position between given s-coordinates of map
            if self.__bool_enable_interpolation:

                # extend localgg array for interpolation
                localgg_extended = np.vstack((localgg_mps2[-2, :], localgg_mps2))

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

                    # Check Neighboring Cells --------------------------------------------------------------------------
                    # check neighboring cell values to always guarantee a conservative acceleration limit

                    # get information of neighbouring data separately for ax and ay
                    bool_idx_isequal_idxplus = localgg_extended[idx] == localgg_extended[idx + 1]
                    bool_idx_greater_idxplus = localgg_extended[idx] > localgg_extended[idx + 1]
                    bool_idx_smaller_idxplus = localgg_extended[idx] < localgg_extended[idx + 1]
                    bool_idxminus_isequal_idx = localgg_extended[idx - 1] == localgg_extended[idx]
                    bool_idxminus_smaller_idx = localgg_extended[idx - 1] < localgg_extended[idx]
                    bool_idxminus_greater_idx = localgg_extended[idx - 1] > localgg_extended[idx]

                    # handle all cases where current value is greater than next value
                    if np.any(bool_idx_greater_idxplus):

                        # handle case where last value is smaller than current value
                        if np.any(np.logical_and(bool_idx_greater_idxplus, bool_idxminus_smaller_idx)):
                            ax_out[idx_ax_out, np.logical_and(bool_idx_greater_idxplus, bool_idxminus_smaller_idx)] = \
                                (1 - grad) \
                                * localgg_extended[idx - 1, np.logical_and(bool_idx_greater_idxplus,
                                                                           bool_idxminus_smaller_idx)] \
                                + grad * localgg_extended[idx + 1, np.logical_and(bool_idx_greater_idxplus,
                                                                                  bool_idxminus_smaller_idx)]

                        # handle case where last value is greater than current value
                        if np.any(np.logical_and(bool_idx_greater_idxplus, bool_idxminus_greater_idx)):
                            ax_out[idx_ax_out, np.logical_and(bool_idx_greater_idxplus, bool_idxminus_greater_idx)] = \
                                (1 - grad) \
                                * localgg_extended[idx, np.logical_and(bool_idx_greater_idxplus,
                                                                       bool_idxminus_greater_idx)] \
                                + grad * localgg_extended[idx + 1, np.logical_and(bool_idx_greater_idxplus,
                                                                                  bool_idxminus_greater_idx)]

                        # handle case where last value is equal to current value
                        if np.any(np.logical_and(bool_idx_greater_idxplus, bool_idxminus_isequal_idx)):
                            ax_out[idx_ax_out, np.logical_and(bool_idx_greater_idxplus, bool_idxminus_isequal_idx)] = \
                                (1 - grad) \
                                * localgg_extended[idx, np.logical_and(bool_idx_greater_idxplus,
                                                                       bool_idxminus_isequal_idx)] \
                                + grad * localgg_extended[idx + 1, np.logical_and(bool_idx_greater_idxplus,
                                                                                  bool_idxminus_isequal_idx)]

                    # handle all cases where current value is smaller than next value
                    if np.any(bool_idx_smaller_idxplus):

                        # handle case where last value is smaller than current value
                        if np.any(np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_smaller_idx)):
                            ax_out[idx_ax_out, np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_smaller_idx)] = \
                                (1 - grad) \
                                * localgg_extended[idx - 1, np.logical_and(bool_idx_smaller_idxplus,
                                                                           bool_idxminus_smaller_idx)] \
                                + grad * localgg_extended[idx, np.logical_and(bool_idx_smaller_idxplus,
                                                                              bool_idxminus_smaller_idx)]

                        # handle case where last value is greater than current value
                        if np.any(np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_greater_idx)):
                            ax_out[idx_ax_out, np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_greater_idx)] = \
                                localgg_extended[idx, np.logical_and(bool_idx_smaller_idxplus,
                                                                     bool_idxminus_greater_idx)]

                        # handle case where last value is equal to current value
                        if np.any(np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_isequal_idx)):

                            ax_out[idx_ax_out, np.logical_and(bool_idx_smaller_idxplus, bool_idxminus_isequal_idx)] = \
                                localgg_extended[idx, np.logical_and(bool_idx_smaller_idxplus,
                                                                     bool_idxminus_isequal_idx)]

                    # handle all cases where current value is equal to next value
                    if np.any(bool_idx_isequal_idxplus):
                        ax_out[idx_ax_out, bool_idx_isequal_idxplus] = localgg_extended[idx, bool_idx_isequal_idxplus]

                    idx_ax_out += 1

                # time_globalpath = time.time() - tic_f
                # logging.debug('time to get local acceleration limits: ' + str(time_globalpath))

            # if False, no interpolation is used; values of the corresponding tpamap section are taken
            else:
                idx_list = []

                version = 2

                if version == 1:

                    for row in s_actual_m:
                        idx = np.argmin(np.abs(self.coordinates_sxy_m[:, 0] - row)) + 0

                        if (self.coordinates_sxy_m_extended[idx + 1, 0] - row) > 0:
                            idx -= 1

                            if idx < 0:
                                idx = self.coordinates_sxy_m.shape[0]

                        idx_list.append(idx)

                    ax_out = localgg_mps2[idx_list, :]

                elif version == 2:
                    i = np.searchsorted(self.coordinates_sxy_m[:, 0], s_actual_m, side='right') - 1
                    i[i < 0] = 0

                    # print(np.max(np.abs(np.asarray(idx_list) - i)))

                    ax_out = localgg_mps2[i, :]

            # if velocity dependence is enabled, the local acceleration limits are interpolated
            if self.__bool_enable_velocitydependence:

                # TEST
                # velocity_mps= np.linspace(1,97,100)[:, np.newaxis]

                # new version implemented due to performance issues
                # version 2 is 2x faster
                version = 2

                if version == 1:

                    ax = []
                    ay = []
                    for i in range(ax_out.shape[0]):
                        ax.append(np.interp(velocity_mps[i], self.velocity_steps[1:], ax_out[i, 0::2]))
                        ay.append(np.interp(velocity_mps[i], self.velocity_steps[1:], ax_out[i, 1::2]))

                elif version == 2:

                    ax, ay = self.interp_velocitysteps(x=velocity_mps,
                                                       xp=np.hstack((self.velocity_steps, 150)),
                                                       fp=np.concatenate((ax_out[:, :2], ax_out, ax_out[:, -2:]),
                                                                         axis=1))

                # print(np.max(np.abs(ax_new - ax)))
                # print(np.max(np.abs(ay_new - ay)))

                localgg_out = np.hstack((ax, ay))

            else:
                localgg_out = ax_out.copy()

        # raise error, if shape of return localgg is not equal to input trajectory; must have same length
        if position_m.shape[0] != localgg_out.shape[0]:
            raise ValueError('TPA MapInterface: number of rows of arrays for position request (input) and localgg '
                             '(output) do not match!')

        return localgg_out

    # ------------------------------------------------------------------------------------------------------------------

    def interp_velocitysteps(self, x: np.array, xp: np.array, fp: np.array):
        """Interpolates acceleration limits for requested velocity steps.

        Input
        :param x:           velocity steps for which the acceleration limits are requested
        :type x: np.array
        :param xp:          velocity steps of the tpa-map (velocity where accelertion limits are provided)
        :type xp: np.array
        :param fp:          acceleration limits used for interpolation
        :type fp: np.array

        Output
        :return:            interpolated acceleration limits
        :rtype: tuple
        """

        # sort input velocity into velocity areas between interpolation points
        j = np.searchsorted(xp, x) - 1
        j[j < 0] = 0

        # get interpolation factor
        d = (x - xp[j]) / (xp[j + 1] - xp[j])

        if fp.ndim == 2:
            fpx_temp = fp[:, 0::2]
            fpy_temp = fp[:, 1::2]
            axis_interp = 1

        elif fp.ndim == 1:
            fpx_temp = fp[0::2]
            fpy_temp = fp[1::2]
            axis_interp = 0

        else:
            raise ValueError('TPA MapInterface: dimension error of array for velocity interpolation!')

        # interpolate ax values
        ax = (1 - d) * np.take_along_axis(fpx_temp, j, axis=axis_interp) \
            + np.take_along_axis(fpx_temp, j + 1, axis=axis_interp) * d

        # interpolate ay values
        ay = (1 - d) * np.take_along_axis(fpy_temp, j, axis=axis_interp) \
            + np.take_along_axis(fpy_temp, j + 1, axis=axis_interp) * d

        return ax, ay

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
                    self.section_id = data_tpainterface[0][:, 0]
                    self.sectionid_change = np.concatenate((np.asarray([True]), np.diff(self.section_id) > 0))

                    self.coordinates_sxy_orignal_m = data_tpainterface[0][:, 1:4]
                    self.coordinates_sxy_m = self.coordinates_sxy_orignal_m[self.sectionid_change]

                    # check if data for velocity steps is available
                    if np.all(data_tpainterface[1]) and len(data_tpainterface[1]) > 0:
                        self.velocity_steps = data_tpainterface[1]
                        self.__count_velocity_steps = int(len(self.velocity_steps))
                        self.velocity_steps = np.hstack(([0.0], self.velocity_steps))
                        self.__bool_enable_velocitydependence = True

                    else:
                        self.velocity_steps = np.zeros(1)
                        self.__count_velocity_steps = 1
                        self.__bool_enable_velocitydependence = False

                    self.format_rawtpamap()

                    self.__bool_received_tpamap = True

                    # if current data mode is global_constant and data is received, switch to global_variable
                    if self.data_mode == 'global_constant':
                        self.localgg_mps2 = np.full((self.coordinates_sxy_m.shape[0], self.__count_velocity_steps * 2),
                                                    np.tile(self.localgg_mps2, self.__count_velocity_steps))
                        self.data_mode = 'global_variable'

                self.__localgg_lastupdate = data_tpainterface[0][:, 4:][self.sectionid_change]

                # insert updates beyond current planning horizon of ltpl
                self.insert_tpa_updates(array_to_update=self.localgg_mps2,
                                        array_data=self.__localgg_lastupdate)

    # ------------------------------------------------------------------------------------------------------------------

    def insert_tpa_updates(self,
                           array_to_update: np.array,
                           array_data: np.array):
        """Inserts local acceleration limits into the localgg array with respect to the planning horizon of the local
           trajectory planner.

        :param array_to_update: [description]
        :type array_to_update: np.array
        :param array_data: [description]
        :type array_data: np.array
        :return: [description]
        :rtype: np.array
        """

        # update stored tpamap besides planning horizon of local trajectory planer
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
                array_to_update[idx_start:, :] = array_data[idx_start:, :]
                array_to_update[:idx_end, :] = array_data[:idx_end, :]

            else:
                array_to_update[idx_start:idx_end, :] = array_data[idx_start:idx_end, :]

            self.localgg_mps2 = array_to_update.copy()

        else:
            return

    # ------------------------------------------------------------------------------------------------------------------

    def set_acclim_strategy(self,
                            ax_strat_mps2: float,
                            ay_strat_mps2: float,
                            bool_isactivate_strategy: bool):
        """Updates the tpa-map with acceleration limits which are set via race control.

        :param ax_strat_mps2: long. acceleration limit from race control
        :type ax_strat_mps2: float
        :param ay_strat_mps2: lat. acceleration limit from race control
        :type ay_strat_mps2: float
        :param bool_isactivate_strategy: flag to indicate whether or not race control target is active
        :type bool_isactivate_strategy: bool
        """

        # check whether strategy limitations are active
        if bool_isactivate_strategy:

            i_rows = self.coordinates_sxy_m.shape[0]
            i_columns = self.__count_velocity_steps

            # np.tile: Construct an array by repeating A the number of times given by reps
            manip_localgg_mps2 = np.tile(np.hstack((np.full((i_rows, 1), ax_strat_mps2),
                                                    np.full((i_rows, 1), ay_strat_mps2))), i_columns)

            self.localgg_strat_mps2 = self.insert_tpa_updates(array_to_update=self.localgg_strat_mps2,
                                                              array_data=manip_localgg_mps2)

        # detect when strategy stops to send acceleration limits
        if self.bool_isactivate_strategy != bool_isactivate_strategy and not bool_isactivate_strategy:
            self.bool_switchedoff_strat = True

            self.localgg_strat_mps2 = self.localgg_mps2.copy()

        else:
            self.bool_switchedoff_strat = False

        # avoid processing strategy input if tpa-map does not contain coordinates of the race track (-> use varloc mode)
        if bool_isactivate_strategy and self.localgg_mps2.shape[0] <= 1:
            self.bool_isactivate_strategy = False
            print('TPA MapInterface: WARNING: strategy input is ignored! intial localgg map must contain variable '
                  'location info!')

        else:
            self.bool_isactivate_strategy = bool_isactivate_strategy

    # ------------------------------------------------------------------------------------------------------------------

    def scale_acclim_lapwise(self,
                             lap_counter: int,
                             laps_interp: list,
                             s_current_m: float,
                             s_total_m: float,
                             scaling: list):
        """Calculates linearly interpolated scaling factor for acceleration limits.

        :param lap_counter:     number of current lap (first lap is expected with 0)
        :type lap_counter:      int
        :param laps_interp:     number of laps for interpolation: first entry = start lap, second entry = end lap
        :type laps_interp:      list
        :param s_current_m:     current position's s-coordinate
        :type s_current_m:      float
        :param s_total_m:       total length of racetrack
        :type s_total_m:        float
        :param scaling:         scale factors: first entry = start scaling, second entry = end scaling
        :type scaling:          list
        """

        self.__scalefactor_tmp = np.interp(lap_counter * s_total_m + s_current_m,
                                           np.asarray(laps_interp) * s_total_m,
                                           scaling)

        # print('lap_counter: {}, s_current_m: {}, scalefactor_tmp: {}'
        #       .format(lap_counter, s_current_m, self.__scalefactor_tmp))


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
