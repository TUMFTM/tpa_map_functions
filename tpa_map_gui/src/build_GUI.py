import numpy as np
import sys
import os.path
import math
import logging
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import tkinter as tk

# testing
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

path2module = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2module)

# custom functions
import tpa_map_functions as tmf

"""
Created by: Dominik Staerk
Created on: 04.11.2020
"""


logging.basicConfig(format='%(levelname)s:IN--%(funcName)s--: %(message)s', level=logging.WARNING)


class Manager(tk.Canvas):
    def __init__(self,
                 filepath2ltpl_refline: str,
                 filepath2output_tpamap: str,
                 stepsize_resample_m: float,
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
        self.filepath2ltpl_refline = filepath2ltpl_refline
        self.filepath2output_tpamap = filepath2output_tpamap
        self.track_name = str(csv_filename)
        self.gui_mode = int(gui_mode)
        self.stepsize_resample_m = float(stepsize_resample_m)
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

        # load reference line
        self.output_data = tmf.helperfuncs.preprocess_ltplrefline.\
            preprocess_ltplrefline(filepath2ltpl_refline=self.filepath2ltpl_refline,
                                   stepsize_resample_m=self.stepsize_resample_m)

        refline_resampled = self.output_data['refline_resampled']['refline_resampled']
        bool_closedtrack = self.output_data['bool_closedtrack']

        # testing
        if self.gui_mode == 2:
            self.ax = np.vstack(
                self.mean_acc - self.amplitude_acc + 2 * self.amplitude_acc * np.random.sample(len(refline_resampled)))
            self.ay = np.vstack(
                self.mean_acc - self.amplitude_acc + 2 * self.amplitude_acc * np.random.sample(len(refline_resampled)))
            self.local_scaling = np.vstack(
                self.mean_acc - self.amplitude_acc + 2 * self.amplitude_acc * np.random.sample(len(refline_resampled)))
        else:
            self.local_scaling_long = np.vstack(
                self.mean_lsc - self.amplitude_lsc + 2 * self.amplitude_lsc * np.random.sample(len(refline_resampled)))
            self.local_scaling_lat = np.vstack(
                self.mean_lsc - self.amplitude_lsc + 2 * self.amplitude_lsc * np.random.sample(len(refline_resampled)))
            self.local_scaling = np.vstack(
                self.mean_lsc - self.amplitude_lsc + 2 * self.amplitude_lsc * np.random.sample(len(refline_resampled)))

        if self.gui_mode == 1:
            self.refline_initial = np.hstack([refline_resampled, self.local_scaling, self.local_scaling_long,
                                              self.local_scaling_lat])
        elif self.gui_mode == 2:
            self.refline_initial = np.hstack([refline_resampled, self.local_scaling, self.ax, self.ay])

        self.bool_closedtrack = bool_closedtrack

        self.refline_manip = self.refline_initial.copy()

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

        # plot figure
        self.plot_tpamap(self.refline_initial)

    # ------------------------------------------------------------------------------------------------------------------
    # Class functions --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def gui_mode_select_1(self):
        self.__init__(
            filepath2ltpl_refline=self.filepath2ltpl_refline,
            filepath2output_tpamap=self.filepath2output_tpamap,
            stepsize_resample_m=self.stepsize_resample_m,
            gui_mode=1,
            csv_filename=self.track_name,
            default=self.default)

    # ------------------------------------------------------------------------------------------------------------------

    def gui_mode_select_2(self):
        self.__init__(
            filepath2ltpl_refline=self.filepath2ltpl_refline,
            filepath2output_tpamap=self.filepath2output_tpamap,
            stepsize_resample_m=self.stepsize_resample_m,
            gui_mode=2,
            csv_filename=self.track_name,
            default=self.default)

    # ------------------------------------------------------------------------------------------------------------------

    def randomize(self):
        refline_resampled = self.output_data['refline_resampled']['refline_resampled']

        if self.gui_mode == 2:
            self.ax = np.vstack(
                self.mean_acc - self.amplitude_acc + 2 * self.amplitude_acc * np.random.sample(len(refline_resampled)))
            self.ay = np.vstack(
                self.mean_acc - self.amplitude_acc + 2 * self.amplitude_acc * np.random.sample(len(refline_resampled)))
            self.local_scaling = np.vstack(
                self.mean_acc - self.amplitude_acc + 2 * self.amplitude_acc * np.random.sample(len(refline_resampled)))

        else:
            self.local_scaling_long = np.vstack(
                self.mean_lsc - self.amplitude_lsc + 2 * self.amplitude_lsc * np.random.sample(len(refline_resampled)))
            self.local_scaling_lat = np.vstack(
                self.mean_lsc - self.amplitude_lsc + 2 * self.amplitude_lsc * np.random.sample(len(refline_resampled)))
            self.local_scaling = np.vstack(
                self.mean_lsc - self.amplitude_lsc + 2 * self.amplitude_lsc * np.random.sample(len(refline_resampled)))

        if self.gui_mode == 1:
            self.refline_initial = np.hstack(
                [refline_resampled, self.local_scaling, self.local_scaling_long, self.local_scaling_lat])
        elif self.gui_mode == 2:
            self.refline_initial = np.hstack([refline_resampled, self.local_scaling, self.ax, self.ay])

        self.update_figure()

    # ------------------------------------------------------------------------------------------------------------------

    def smooth(self):
        refline_resampled = self.output_data['refline_resampled']['refline_resampled']
        random = np.random.rand() * 2 * math.pi

        for row in range(len(refline_resampled)):
            if self.gui_mode == 2:
                self.ax[row] = self.mean_acc + self.amplitude_acc * math.sin(
                    (row / len(refline_resampled)) * 4 * math.pi + random)
                self.ay[row] = self.mean_acc + self.amplitude_acc * math.sin(
                    (row / len(refline_resampled)) * 4 * math.pi + random)
                self.local_scaling[row] = self.mean_acc + self.amplitude_acc * math.sin(
                    (row / len(refline_resampled)) * 4 * math.pi + random)
            else:
                self.local_scaling_long[row] = self.mean_lsc + self.amplitude_lsc * math.sin(
                    (row / len(refline_resampled)) * 4 * math.pi + random)
                self.local_scaling_lat[row] = self.mean_lsc + self.amplitude_lsc * math.sin(
                    (row / len(refline_resampled)) * 4 * math.pi + random)
                self.local_scaling[row] = self.mean_lsc + self.amplitude_lsc * math.sin(
                    (row / len(refline_resampled)) * 4 * math.pi + random)

        if self.gui_mode == 1:
            self.refline_initial = np.hstack(
                [refline_resampled, self.local_scaling, self.local_scaling_long, self.local_scaling_lat])
        elif self.gui_mode == 2:
            self.refline_initial = np.hstack([refline_resampled, self.local_scaling, self.ax, self.ay])

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
                                                             coordinates_sxy_m=refline_plot[:, :3],
                                                             long_limit=self.ax,
                                                             lat_limit=self.ay,
                                                             header_info=header_custom,
                                                             track_name=self.track_name)

        else:
            tmf.helperfuncs.save_tpamap.save_tpamap_fromfile(filepath2output_tpamap=self.filepath2output_tpamap_save,
                                                             coordinates_sxy_m=refline_plot[:, :3],
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
                    tk.Label(self.frame_usermessage, text='make sure row ' + str(int_rows) +
                                                          ' is either completely filled or empty! '
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
        print('get_data done')

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
                        break

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
                        break

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

        local_scaling = refline_plot[:, 3]
        refline_resampled = refline_plot[:, :6]

        # generate main plot -------------------------------------------------------------------------------------------

        if self.fig_handle:
            self.fig_handle.clf()
        else:
            self.fig_handle = plt.figure(figsize=(12, 9))

        self.fig_handle.canvas.manager.window.wm_geometry("+%d+%d" % (600, 10))

        plt.subplot(3, 1, (1, 2))

        plt.plot(refline_resampled[:, 1], refline_resampled[:, 2], 'k', label='resampled reference line')

        plotting_distance_m = np.arange(0, refline_resampled[-1, 0], 100)

        for int_counter, ele in enumerate(plotting_distance_m):
            array = np.asarray(refline_resampled[:, 0])
            idx = (np.abs(array - ele)).argmin()

            plt.plot(refline_resampled[idx, 1], refline_resampled[idx, 2], 'bx')
            plt.annotate('s=' + str(plotting_distance_m.tolist()[int_counter]) + ' m',
                         (refline_resampled[idx, 1], refline_resampled[idx, 2]),
                         xytext=(0, 30), textcoords='offset points', ha='center', va='bottom', color='blue',
                         bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', color='b'))

        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be numlines x points per line x 2 (x and y)
        points = refline_resampled[:, 1:3].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create the line collection object, setting the colormapping parameters.
        # Have to set the actual values used for colormapping separately.
        lc = LineCollection(segments)
        lc.set_array(local_scaling)
        lc.set_linewidth(7)

        plt.gca().add_collection(lc)

        cbar = self.fig_handle.colorbar(lc)  # (lc, cmap='viridis')
        if self.gui_mode == 2:
            cbar.set_label('local combined acceleration limit mps2', rotation=270)
        else:
            cbar.set_label('local mue scaling factor', rotation=270)

        plt.axis('equal')

        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('x in meters')
        plt.ylabel('y in meters')
        plt.show(block=False)

        plt.subplot(3, 1, 3)

        plt.plot(refline_resampled[:-1, 0], refline_resampled[:-1, 4], 'r', label='longitudinal')
        plt.plot(refline_resampled[:-1, 0], refline_resampled[:-1, 5], 'b', label='lateral')

        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('s in meters')
        if self.gui_mode == 2:
            plt.ylabel('local acc limit in mps2')
        else:
            plt.ylabel('local scaling factor')
        plt.show(block=False)

        print('plot_tpamap done')


# ----------------------------------------------------------------------------------------------------------------------
# testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
