import numpy as np
import sys
import os.path
import math
import logging
from scipy.interpolate import interp1d # noqa F401
import matplotlib.pyplot as plt
import tkinter as tk


# import custom modules
path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions as tmf

"""
Created by: Dominik Staerk
Created on: 04.11.2020
"""

logging.basicConfig(format='%(levelname)s:IN--%(funcName)s--: %(message)s', level=logging.WARNING)


class Manager(tk.Canvas):
    def __init__(self,
                 refline_dict: dict(),
                 refline_resampled: np.array,
                 bool_closedtrack: bool,
                 filepath2output_tpamap: str,
                 gui_mode: int,
                 csv_filename: str,
                 default: dict = dict(),
                 master=None,
                 **kwargs):
        """
        Documentation:          user-interface for setting local gg-scale factors manually

        Input
        :param master:
        :param kwargs:
        """

        if refline_dict['refline_resampled']['section_id'][0, 0] > 1:
            self.refline_resampled = refline_resampled[refline_dict['refline_resampled']['sectionid_change']]
            self.__mode_stepsize = 'var_steps'

        else:
            self.refline_resampled = refline_resampled
            self.__mode_stepsize = 'const_steps'

        if gui_mode == 1 and self.__mode_stepsize == 'var_steps':
            plt.close('all')
            raise ValueError('When GUI is operated with friction coeff., only const. step size is allowed!')

        self.__section_id = refline_dict['refline_resampled']['section_id']

        self.refline_resampled_org = refline_resampled
        self.refline_dict = refline_dict
        self.refline_original = refline_dict['refline']
        self.bool_closedtrack = bool_closedtrack

        self.filepath2output_tpamap = filepath2output_tpamap
        self.track_name = str(csv_filename)
        self.gui_mode = int(gui_mode)
        self.default = default
        self.mean_lsc = default['mean_lsc']
        self.mean_acc = default['mean_acc']
        self.amplitude_lsc = default['amplitude_lsc']
        self.amplitude_acc = default['amplitude_acc']

        tk.Canvas.__init__(self, master, **kwargs)

        # set initial number of rows and columns -----------------------------------------------------------------------
        self.int_number_rows = 8

        if self.gui_mode == 1 or self.gui_mode == 2:
            self.int_number_columns = 4
        else:
            print('Error: invalid GUI mode selected, check settings.ini\nreset to default GUI mode 1')
            self.gui_mode = 1

        self.row_counter = 0
        self.entry_counter = 0
        self.entries = {}
        self.fig_handle = None

        # set up main gui ----------------------------------------------------------------------------------------------

        self.frame_introtext = tk.Frame(master)
        self.frame_introtext.grid(row=0, columnspan=1)

        text_intro = 'abcabc\nxyz\ntesttest'
        tk.Label(self.frame_introtext, text=text_intro)
        # # msg.config(font=('times', 24))

        self.frame_scrollbarlabels = tk.Frame(master)
        self.frame_scrollbarlabels.grid(row=2)

        if self.gui_mode == 2:
            tk.Label(self.frame_scrollbarlabels, text="start of section\nin meters", relief='groove', width=15) \
                .pack(side="left")  # .grid(row=0, column=0, ipadx=5, ipady=10)
            tk.Label(self.frame_scrollbarlabels, text="end of section\nin meters", relief='groove', width=15) \
                .pack(side="left")  # .grid(row=0, column=1, ipadx=5, ipady=10)
            tk.Label(self.frame_scrollbarlabels, text="ax limits", relief='groove', width=15) \
                .pack(side="left")  # .grid(row=0, column=2, ipadx=5, ipady=10)
            tk.Label(self.frame_scrollbarlabels, text="ay limits", relief='groove', width=15) \
                .pack(side="left")  # .grid(row=0, column=2, ipadx=5, ipady=10)
        else:
            tk.Label(self.frame_scrollbarlabels, text="start of section\nin meters", relief='groove', width=15) \
                .pack(side="left")  # .grid(row=0, column=0, ipadx=5, ipady=10)
            tk.Label(self.frame_scrollbarlabels, text="end of section\nin meters", relief='groove', width=15) \
                .pack(side="left")  # .grid(row=0, column=1, ipadx=5, ipady=10)
            tk.Label(self.frame_scrollbarlabels, text="local scaling\nlong", relief='groove', width=15) \
                .pack(side="left")  # .grid(row=0, column=2, ipadx=5, ipady=10)
            tk.Label(self.frame_scrollbarlabels, text="local scaling\nlat", relief='groove', width=15) \
                .pack(side="left")  # .grid(row=0, column=2, ipadx=5, ipady=10)
        # self.frame_scrollbarlabels.columnconfigure(0, weight=1)
        # self.frame_scrollbarlabels.columnconfigure(1, weight=1)
        # self.frame_scrollbarlabels.columnconfigure(2, weight=1)

        self.container = tk.Frame(master)
        self.container.grid(row=3, column=0, sticky='nsew')

        self.canvas = tk.Canvas(self.container, width=500, height=250)
        self.scrollbar = tk.Scrollbar(self.container, orient='vertical', command=self.canvas.yview)
        self.canvas.config(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.scrollbar.grid(row=0, column=1, sticky='nse')

        # self.container.bind('<Configure>')

        for int_counter in range(self.int_number_rows):
            self.addblock()

        self.frame_buttons = tk.Frame(master)
        self.frame_buttons.grid(row=4)

        b1 = tk.Button(self.frame_buttons, text='add new entry', command=self.addblock)
        b1.pack(side='left', padx=5, pady=5)
        b2 = tk.Button(self.frame_buttons, text='update map', command=self.update_figure)
        b2.pack(side='left', padx=5, pady=5)
        b3 = tk.Button(self.frame_buttons, text="reset map", command=self.reset_figure)
        b3.pack(side='left', padx=5, pady=5)
        b5 = tk.Button(self.frame_buttons, text='quit', command=self.quit_job)
        b5.pack(side='left', padx=5, pady=5)

        self.mode_buttons = tk.Frame(master)
        self.mode_buttons.grid(row=5)

        b6 = tk.Button(self.mode_buttons, text='GUI mode 1', command=self.gui_mode_select_1)
        b6.pack(side='left', padx=5, pady=5)
        b7 = tk.Button(self.mode_buttons, text='GUI mode 2', command=self.gui_mode_select_2)
        b7.pack(side='left', padx=5, pady=5)
        b8 = tk.Button(self.mode_buttons, text='randomize map', command=self.randomize)
        b8.pack(side='left', padx=5, pady=5)
        b8 = tk.Button(self.mode_buttons, text='smooth random map', command=self.smooth)
        b8.pack(side='left', padx=5, pady=5)

        self.frame_usermessage = tk.Frame(master)
        self.frame_usermessage.grid(row=6)
        self.frame_usermessage.columnconfigure(0, weight=2)
        self.frame_usermessage.columnconfigure(1, weight=1)

        tk.Label(self.frame_usermessage, text='message: ', relief='groove', width=20).grid(row=0, column=0)
        self.label_usermsg = tk.Label(self.frame_usermessage, text='---').grid(row=0, column=1)

        self.name_prependix = tk.Frame(master)
        self.name_prependix.grid(row=7)

        tk.Label(self.name_prependix,
                 text='enter tpamap_name-pendix (optional): ', relief='groove', width=30).pack(side="left")
        e1 = tk.Entry(self.name_prependix, width=10)
        self.e1 = e1
        e1.pack(side='left', padx=5, pady=5)
        b4 = tk.Button(self.name_prependix, text="save map", command=self.save_map)
        b4.pack(side='left', padx=5, pady=5)

        # intialize tpa map with random values
        self.randomize()

    # ------------------------------------------------------------------------------------------------------------------
    # Class functions --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def gui_mode_select_1(self):
        self.__init__(refline_dict=self.refline_dict,
                      refline_resampled=self.refline_resampled_org,
                      bool_closedtrack=self.bool_closedtrack,
                      filepath2output_tpamap=self.filepath2output_tpamap,
                      gui_mode=1,
                      csv_filename=self.track_name,
                      default=self.default)

    # ------------------------------------------------------------------------------------------------------------------

    def gui_mode_select_2(self):
        self.__init__(refline_dict=self.refline_dict,
                      refline_resampled=self.refline_resampled_org,
                      bool_closedtrack=self.bool_closedtrack,
                      filepath2output_tpamap=self.filepath2output_tpamap,
                      gui_mode=2,
                      csv_filename=self.track_name,
                      default=self.default)

    # ------------------------------------------------------------------------------------------------------------------

    def randomize(self):

        if self.gui_mode == 2:
            self.ax = np.vstack(self.mean_acc
                                - self.amplitude_acc
                                + 2 * self.amplitude_acc * np.random.sample(len(self.refline_resampled)))

            self.ay = np.vstack(self.mean_acc
                                - self.amplitude_acc
                                + 2 * self.amplitude_acc * np.random.sample(len(self.refline_resampled)))

            self.local_scaling = np.vstack(self.mean_acc
                                           - self.amplitude_acc
                                           + 2 * self.amplitude_acc * np.random.sample(len(self.refline_resampled)))

        else:
            self.local_scaling_long = np.vstack(self.mean_lsc
                                                - self.amplitude_lsc
                                                + 2 * self.amplitude_lsc
                                                * np.random.sample(len(self.refline_resampled)))

            self.local_scaling_lat = np.vstack(self.mean_lsc
                                               - self.amplitude_lsc
                                               + 2 * self.amplitude_lsc * np.random.sample(len(self.refline_resampled)))

            self.local_scaling = np.vstack(self.mean_lsc
                                           - self.amplitude_lsc
                                           + 2 * self.amplitude_lsc * np.random.sample(len(self.refline_resampled)))

        if self.gui_mode == 1:
            self.refline_initial = np.hstack([self.refline_resampled,
                                              self.local_scaling,
                                              self.local_scaling_long,
                                              self.local_scaling_lat])
        elif self.gui_mode == 2:
            self.refline_initial = np.hstack([self.refline_resampled,
                                              self.local_scaling,
                                              self.ax,
                                              self.ay])

        self.update_figure()

    # ------------------------------------------------------------------------------------------------------------------

    def smooth(self):
        random = np.random.rand() * 2 * math.pi

        for row in range(len(self.refline_resampled)):
            if self.gui_mode == 2:
                self.ax[row] = self.mean_acc + self.amplitude_acc * math.sin(
                    (row / len(self.refline_resampled)) * 4 * math.pi + random)
                self.ay[row] = self.mean_acc + self.amplitude_acc * math.sin(
                    (row / len(self.refline_resampled)) * 4 * math.pi + random)
                self.local_scaling[row] = self.mean_acc + self.amplitude_acc * math.sin(
                    (row / len(self.refline_resampled)) * 4 * math.pi + random)
            else:
                self.local_scaling_long[row] = self.mean_lsc + self.amplitude_lsc * math.sin(
                    (row / len(self.refline_resampled)) * 4 * math.pi + random)
                self.local_scaling_lat[row] = self.mean_lsc + self.amplitude_lsc * math.sin(
                    (row / len(self.refline_resampled)) * 4 * math.pi + random)
                self.local_scaling[row] = self.mean_lsc + self.amplitude_lsc * math.sin(
                    (row / len(self.refline_resampled)) * 4 * math.pi + random)

        if self.gui_mode == 1:
            self.refline_initial = np.hstack(
                [self.refline_resampled, self.local_scaling, self.local_scaling_long, self.local_scaling_lat])
        elif self.gui_mode == 2:
            self.refline_initial = np.hstack([self.refline_resampled, self.local_scaling, self.ax, self.ay])

        self.update_figure()

    # ------------------------------------------------------------------------------------------------------------------

    def quit_job(self):
        exit()

    # ------------------------------------------------------------------------------------------------------------------

    def addblock(self):
        self.frame_entries = tk.Frame(self.canvas)
        self.frame_entries.pack(anchor='center')

        for column in range(self.int_number_columns):
            entry = tk.Entry(self.frame_entries, width=15)
            self.entries[self.entry_counter] = entry
            entry.grid(row=0, column=column)
            self.entry_counter += 1

        self.frame_entries.columnconfigure(0, weight=1)
        self.frame_entries.columnconfigure(1, weight=1)
        self.frame_entries.columnconfigure(2, weight=1)

        self.canvas.create_window((0, (self.row_counter * 25)),
                                  window=self.frame_entries,
                                  anchor="nw",
                                  width=450,
                                  height=24)

        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        self.row_counter += 1
        print('add_block done')

    # ------------------------------------------------------------------------------------------------------------------

    def print_stuff(self, refline_plot: np.array):

        header_custom = {'track': self.track_name, 'gui_mode': self.gui_mode}

        if not self.entries[0].get():
            print('empty')

        else:
            print('not empty: ' + str(self.entries[0].get()))

        if not self.e1.get():
            print('pendix empty')
            self.filepath2output_tpamap_save = self.filepath2output_tpamap + '.csv'

        else:
            print('pendix: ' + str(self.e1.get()))
            self.filepath2output_tpamap_save = self.filepath2output_tpamap + '_' + self.e1.get() + '.csv'

        if self.gui_mode == 2:
            tmf.helperfuncs.save_tpamap.save_tpamap_fromfile(filepath2output_tpamap=self.filepath2output_tpamap_save,
                                                             mode_save_tpamap='acclimits',
                                                             coordinates_sxy_m=self.refline_resampled_org,
                                                             long_limit=self.ax,
                                                             lat_limit=self.ay,
                                                             section_id=self.__section_id,
                                                             header_info=header_custom,
                                                             track_name=self.track_name)

        else:
            tmf.helperfuncs.save_tpamap.save_tpamap_fromfile(filepath2output_tpamap=self.filepath2output_tpamap_save,
                                                             mode_save_tpamap='frictioncoeff',
                                                             coordinates_sxy_m=self.refline_resampled_org,
                                                             long_limit=self.local_scaling_long,
                                                             lat_limit=self.local_scaling_lat,
                                                             header_info=header_custom,
                                                             track_name=self.track_name)

        print('print_stuff done')

    # ------------------------------------------------------------------------------------------------------------------

    def get_data(self) -> np.array:

        self.array_guidata = np.zeros([self.row_counter, self.int_number_columns])

        int_rows = 0
        int_columns = 0

        for entry in self.entries.values():

            if entry.get():
                try:
                    entry = float(entry.get())
                    self.array_guidata[int_rows, int_columns] = entry

                except ValueError:
                    logging.warning('entries must not be strings!')
                    break

            else:
                self.array_guidata[int_rows, int_columns] = np.nan

            # check values of one entry row for validity
            if int_columns == 3 and self.gui_mode == 1:

                if 0 < np.isnan(self.array_guidata[int_rows, :]).sum() < self.int_number_columns:
                    tk.Label(self.frame_usermessage,
                             text='make sure row ' + str(int_rows) + ' is either completely filled or empty! '
                             'Please double-check').grid(row=0, column=1, sticky=tk.W)

                    break

                elif not np.isnan(self.array_guidata[int_rows, :]).any():

                    if (self.array_guidata[int_rows, :2] < 0).any():
                        logging.warning('at least one track coordinate values in row ' + str(int_rows)
                                        + ' is < 0! Please double-check')
                        break

                    elif self.array_guidata[int_rows, 2:3] <= 0:
                        logging.warning('scaling factor in row ' + str(int_rows)
                                        + ' is <= 0! Please double-check')
                        break

                    elif self.array_guidata[int_rows, 0] >= self.array_guidata[int_rows, 1]:
                        logging.warning('second value in row ' + str(int_rows)
                                        + ' is <= first value! Please double-check')
                        break

                    self.label_usermsg = tk.Label(self.frame_usermessage, text='data successfully updated') \
                        .grid(row=0, column=1, sticky=tk.W)

            elif int_columns == 3 and self.gui_mode == 2:

                if 0 < np.isnan(self.array_guidata[int_rows, :]).sum() < self.int_number_columns:
                    tk.Label(self.frame_usermessage,
                             text='make sure row ' + str(int_rows) + ' is either completely filled or empty!'
                             'Please double-check').grid(row=0, column=1, sticky=tk.W)

                    break

                elif not np.isnan(self.array_guidata[int_rows, :]).any():

                    if (self.array_guidata[int_rows, :2] < 0).any():
                        logging.warning('at least one track coordinate values in row ' + str(int_rows)
                                        + ' is < 0! Please double-check')
                        break

                    elif self.array_guidata[int_rows, 2:3] <= 0:
                        logging.warning('acc limit in row ' + str(int_rows)
                                        + ' is <= 0! Please double-check')
                        break

                    elif self.array_guidata[int_rows, 0] >= self.array_guidata[int_rows, 1]:
                        logging.warning('second value in row ' + str(int_rows)
                                        + ' is <= first value! Please double-check')
                        break

                    self.label_usermsg = tk.Label(self.frame_usermessage, text='data successfully updated') \
                        .grid(row=0, column=1, sticky=tk.W)

            int_columns += 1

            if int_columns == self.int_number_columns:
                int_rows += 1
                int_columns = 0

    # ------------------------------------------------------------------------------------------------------------------

    def update_figure(self):

        self.refline_manip = self.refline_initial.copy()

        self.get_data()

        self.resample_userdata()

        self.plot_tpamap(self.refline_manip)
        print('update_figure done')

    # ------------------------------------------------------------------------------------------------------------------

    def save_map(self):

        self.refline_manip = self.refline_initial.copy()

        self.get_data()

        self.resample_userdata()

        self.print_stuff(self.refline_manip)

        self.plot_tpamap(self.refline_manip)

        tk.Label(self.frame_usermessage, text='map successfully saved').grid(row=0, column=1, sticky=tk.W)

        print('save_map done')

    # ------------------------------------------------------------------------------------------------------------------

    def resample_userdata(self):
        '''
        resample input data to the given reference line coordinates and stepsizes
        '''
        if self.gui_mode == 2:
            for int_rows in range(self.row_counter):

                if not np.isnan(self.array_guidata[int_rows, :]).any():
                    s_start = self.array_guidata[int_rows, 0]
                    s_end = self.array_guidata[int_rows, 1]
                    idx_start = self.find_nearest(self.refline_initial[:, 0], s_start)
                    idx_end = self.find_nearest(self.refline_initial[:, 0], s_end)
                    self.ax[idx_start:idx_end] = self.array_guidata[int_rows, 2]
                    self.ay[idx_start:idx_end] = self.array_guidata[int_rows, 3]
                    a_combined = (self.ax[idx_start] + self.ay[idx_start]) / 2
                    if s_end >= self.refline_manip[-1, 0]:
                        self.ax[-1] = self.array_guidata[int_rows, 2]
                        self.ay[-1] = self.array_guidata[int_rows, 3]
                        a_combined = (self.ax[idx_start] + self.ay[idx_start]) / 2

                    if idx_end == idx_start or ((idx_end < idx_start) and not self.bool_closedtrack):
                        continue

                    elif (idx_end < idx_start) and self.bool_closedtrack:
                        self.refline_manip[idx_end:idx_start, 3] = a_combined
                        self.refline_manip[idx_end:idx_start, 4] = self.ax[idx_start]
                        self.refline_manip[idx_end:idx_start, 5] = self.ay[idx_start]

                    else:
                        self.refline_manip[idx_start:idx_end, 3] = a_combined
                        self.refline_manip[idx_start:idx_end, 4] = self.ax[idx_start]
                        self.refline_manip[idx_start:idx_end, 5] = self.ay[idx_start]

                else:
                    continue
        else:
            for int_rows in range(self.row_counter):

                if not np.isnan(self.array_guidata[int_rows, :]).any():
                    s_start = self.array_guidata[int_rows, 0]
                    s_end = self.array_guidata[int_rows, 1]
                    idx_start = self.find_nearest(self.refline_initial[:, 0], s_start)
                    idx_end = self.find_nearest(self.refline_initial[:, 0], s_end)
                    self.local_scaling_long[idx_start:idx_end] = self.array_guidata[int_rows, 2]
                    self.local_scaling_lat[idx_start:idx_end] = self.array_guidata[int_rows, 3]
                    local_scaling = (self.local_scaling_long[idx_start] + self.local_scaling_lat[idx_start]) / 2
                    if s_end >= self.refline_manip[-1, 0]:
                        self.local_scaling_long[-1] = self.array_guidata[int_rows, 2]
                        self.local_scaling_lat[-1] = self.array_guidata[int_rows, 3]
                        local_scaling = (self.local_scaling_long[idx_start] + self.local_scaling_lat[idx_start]) / 2

                    if idx_end == idx_start or ((idx_end < idx_start) and not self.bool_closedtrack):
                        continue

                    elif (idx_end < idx_start) and self.bool_closedtrack:
                        self.refline_manip[idx_end:idx_start, 3] = local_scaling
                        self.refline_manip[idx_end:idx_start, 4] = self.local_scaling_long[idx_start]
                        self.refline_manip[idx_end:idx_start, 5] = self.local_scaling_lat[idx_start]

                    else:
                        self.refline_manip[idx_start:idx_end, 3] = local_scaling
                        self.refline_manip[idx_start:idx_end, 4] = self.local_scaling_long[idx_start]
                        self.refline_manip[idx_start:idx_end, 5] = self.local_scaling_lat[idx_start]

                else:
                    continue
        print('resample_userdata done')

    # ------------------------------------------------------------------------------------------------------------------

    def reset_figure(self):

        self.refline_manip = self.refline_initial.copy()

        self.plot_tpamap(self.refline_initial)
        print('reset_figure done')

    # ------------------------------------------------------------------------------------------------------------------

    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return idx

    # ------------------------------------------------------------------------------------------------------------------

    def plot_tpamap(self, refline_plot: np.array):

        tpamap = np.hstack((refline_plot[:, :3], refline_plot[:, 4:]))

        # if self.gui_mode == 2:
        #     ylabel = 'local acc. limit in m/s^2'
        # else:
        #     ylabel = 'local scaling factor'

        # dict_plotinfo = {'bool_set_blocking': False,
        #                  'ylabel': ylabel}

        # generate main plot -------------------------------------------------------------------------------------------

        if self.fig_handle:
            self.fig_handle.clf()
        else:
            self.fig_handle = plt.figure(figsize=(12, 9))

        self.fig_handle.canvas.manager.window.wm_geometry("+%d+%d" % (600, 10))

        tmf.visualization.visualize_tpamap.visualize_tpamap(tpamap=tpamap,
                                                            refline=self.refline_dict['refline'],
                                                            width_right=self.refline_dict['width_right'],
                                                            width_left=self.refline_dict['width_left'],
                                                            normvec_normalized=self.refline_dict['normvec_normalized'],
                                                            fig_handle=self.fig_handle)

        print('plot_tpamap done')


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
