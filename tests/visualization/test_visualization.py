import os.path
import sys

# import custom modules
path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions as tmf

# User Input -------------------------------------------------------------------------------------------------------

track_name = 'berlin'
tpamap_name = 'tpamap_varloc_varvel_berlin'
bool_enable_debug = True

# Preprocess Reference Line ----------------------------------------------------------------------------------------

filepath2ltpl_refline = os.path.join(path2tmf, 'inputs', 'traj_ltpl_cl', 'traj_ltpl_cl_' + track_name + '.csv')
filepath2tpamap = os.path.join(path2tmf, 'outputs', tpamap_name + '.csv')

dict_output = tmf.helperfuncs.preprocess_ltplrefline.preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline,
                                                                            bool_enable_debug=bool_enable_debug)

tmf.visualization.visualize_tpamap.visualize_tpamap(filepath2tpamap=filepath2tpamap,
                                                    refline=dict_output['refline'],
                                                    width_right=dict_output['width_right'],
                                                    width_left=dict_output['width_left'],
                                                    normvec_normalized=dict_output['normvec_normalized'])
