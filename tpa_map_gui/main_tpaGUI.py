import os.path
import src
import tkinter as tk


# User Input -----------------------------------------------------------------------------------------------------------

# csv_filename - trackname in inputs-folder
name_refline = "berlin"

# stepsize_resample_m - desired stepsize for tpa-map resolution
stepsize_resample_m = 10

# gui_mode = 1: mode to customize local scaling factor for racetrack sections
# gui_mode = 2: mode to customize ax and ay limits for racetrack sections
gui_mode = 1

dict_settings = {"mean_lsc": 1.0,       # mean of the random created local scaling factors
                 "mean_acc": 12.0,      # mean of the random created acceleration limits
                 "amplitude_lsc": 0.5,  # amplitude of the random created local scaling factors around the mean
                 "amplitude_acc": 1.0}  # amplitude of the random created acceleration limits around the mean


# Path Management ------------------------------------------------------------------------------------------------------

path2module = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')

filepath2ltpl_refline = os.path.join(path2module, 'inputs', 'traj_ltpl_cl',
                                     'traj_ltpl_cl_' + name_refline + '.csv')

filepath2output_tpamap = os.path.join(path2module, 'outputs', 'tpamap_' + name_refline)

tk_root = tk.Tk()
tk_root.title("Settings for local gg-scaling")
tk_root.geometry('%dx%d+%d+%d' % (550, 450, 10, 10))

manager = src.build_GUI.Manager(master=tk_root,
                                filepath2ltpl_refline=filepath2ltpl_refline,
                                filepath2output_tpamap=filepath2output_tpamap,
                                stepsize_resample_m=stepsize_resample_m,
                                gui_mode=gui_mode,
                                csv_filename=name_refline,
                                default=dict_settings)

tk_root.mainloop()
