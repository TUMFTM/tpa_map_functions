import os.path
import src
import sys
import tkinter as tk

# import custom modules
path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions as tmf


# User Input -----------------------------------------------------------------------------------------------------------

# csv_filename - trackname in inputs-folder
name_refline = "berlin"

# stepsize_resample_m - desired stepsize for tpa-map resolution
stepsize_resample_m = 25

mode_resample_refline = 'var_steps'
section_length_min_m = 15
section_length_max_m = 200

# gui_mode = 1: mode to customize local scaling factor for racetrack sections
# gui_mode = 2: mode to customize ax and ay limits for racetrack sections
gui_mode = 2

dict_settings = {"mean_lsc": 1.0,       # mean of the random created local scaling factors
                 "mean_acc": 12.0,      # mean of the random created acceleration limits
                 "amplitude_lsc": 0.5,  # amplitude of the random created local scaling factors around the mean
                 "amplitude_acc": 1.0}  # amplitude of the random created acceleration limits around the mean


# Path Management ------------------------------------------------------------------------------------------------------

filepath2ltpl_refline = os.path.join(path2tmf, 'inputs', 'traj_ltpl_cl', 'traj_ltpl_cl_' + name_refline + '.csv')

filepath2output_tpamap = os.path.join(path2tmf, 'outputs', 'tpamap_' + name_refline)


# Load reference line --------------------------------------------------------------------------------------------------

refline_dict = tmf.helperfuncs.preprocess_ltplrefline.\
    preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline,
                           mode_resample_refline=mode_resample_refline,
                           stepsize_resample_m=stepsize_resample_m,
                           section_length_limits_m=[section_length_min_m, section_length_max_m])

# Set Up GUI -----------------------------------------------------------------------------------------------------------

tk_root = tk.Tk()
tk_root.title("Settings for local gg-scaling")
tk_root.geometry('%dx%d+%d+%d' % (550, 450, 10, 10))

manager = src.build_GUI.Manager(master=tk_root,
                                refline=refline_dict['refline_resampled']['refline_resampled'],
                                bool_closedtrack=refline_dict['bool_closedtrack'],
                                filepath2output_tpamap=filepath2output_tpamap,
                                gui_mode=gui_mode,
                                csv_filename=name_refline,
                                default=dict_settings)

tk_root.mainloop()
