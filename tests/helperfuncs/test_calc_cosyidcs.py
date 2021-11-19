import os.path
import sys
import numpy as np

# import custom modules
path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions as tmf

# User Input -------------------------------------------------------------------------------------------------------

track_name = 'modena'
bool_enable_debug = True

# mode for resampling reference line, options: "const_steps", "var_steps"
mode_resample_refline = 'var_steps'
stepsize_resample_m = 11.11
section_length_min_m = 15
section_length_max_m = 200

# test data
s_coordinates = np.asarray([239.09252732, 239.30584717, 239.51885986, 239.73167419, 239.94430542, 240.15682983,
                            240.36885071])

# Preprocess Reference Line ----------------------------------------------------------------------------------------

filepath2ltpl_refline = os.path.join(path2tmf, 'inputs', 'traj_ltpl_cl', 'traj_ltpl_cl_' + track_name + '.csv')


output_data = tmf.helperfuncs.preprocess_ltplrefline.\
    preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline,
                           mode_resample_refline=mode_resample_refline,
                           stepsize_resample_m=stepsize_resample_m,
                           section_length_limits_m=[section_length_min_m, section_length_max_m],
                           bool_enable_debug=bool_enable_debug)

refline_resampled = output_data["refline_resampled"]

test_output = tmf.helperfuncs.calc_cosyidcs.calc_cosyidcs(np.squeeze(refline_resampled["section_id"]),
                                                          refline_resampled["refline_resampled"],
                                                          s_coordinates)

print(test_output)
