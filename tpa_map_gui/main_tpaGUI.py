import os.path
import sys
import tkinter as tk

# import custom modules
path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions as tmf
import tpa_map_gui.src.build_GUI as build_GUI


# User Input -----------------------------------------------------------------------------------------------------------

# track name (located in /inputs/traj_ltpl_cl/traj_ltpl_cl_[track name].csv)
name_refline = "berlin"

# mode for reference line resampling - options "const_steps", "var_steps"
mode_resample_refline = 'var_steps'

# if "const_steps": stepsize_resample_m - desired stepsize for tpa-map resolution
stepsize_resample_m = 25

# if "var_steps":
#   - section_length_min_m: min. section length
#   - section_length_max_m: max. section length
section_length_min_m = 15
section_length_max_m = 200

# gui_mode
#   1: mode to customize local tire scaling factors
#   2: mode to customize local acceleration limits
gui_mode = 2

# optional: necessary for initialization and for randomizing
dict_settings = {"mean_lsc": 1.0,       # mean of the random created local scaling factors
                 "mean_acc": 12.0,      # mean of the random created acceleration limits
                 "amplitude_lsc": 0.5,  # amplitude of the random created local scaling factors around the mean
                 "amplitude_acc": 1.0}  # amplitude of the random created acceleration limits around the mean

# Manage paths ---------------------------------------------------------------------------------------------------------

filepath2ltpl_refline = os.path.join(path2tmf, 'inputs', 'traj_ltpl_cl', 'traj_ltpl_cl_' + name_refline + '.csv')

filepath2output_tpamap = os.path.join(path2tmf, 'outputs', 'tpamap_' + name_refline)

# Load reference line --------------------------------------------------------------------------------------------------

refline_dict = tmf.helperfuncs.preprocess_ltplrefline.\
    preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline,
                           mode_resample_refline=mode_resample_refline,
                           stepsize_resample_m=stepsize_resample_m,
                           section_length_limits_m=[section_length_min_m, section_length_max_m])

# Set up GUI -----------------------------------------------------------------------------------------------------------

tk_root = tk.Tk()
tk_root.title("Settings for local gg-scaling")
tk_root.geometry('%dx%d+%d+%d' % (550, 450, 10, 10))

manager = build_GUI.Manager(master=tk_root,
                            refline_dict=refline_dict,
                            refline_resampled=refline_dict['refline_resampled']['refline_resampled'],
                            bool_closedtrack=refline_dict['bool_closedtrack'],
                            filepath2output_tpamap=filepath2output_tpamap,
                            gui_mode=gui_mode,
                            csv_filename=name_refline,
                            default=dict_settings)

tk_root.mainloop()
