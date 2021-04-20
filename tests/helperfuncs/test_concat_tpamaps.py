import os.path
import sys

# import custom modules
path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions as tmf


# User Input -----------------------------------------------------------------------------------------------------------

# mode = 'local scaling factors'
mode = 'local acceleration limits'

# settings for mode 'local acceleration limits'
if mode == 'local acceleration limits':

    filename_output = 'tpamap_varloc_varvel.csv'

    # part of naming which is identical for all tpa-maps which should be concatenated; only difference is "_XXmps"
    tpamap_identifier = "tpamap_berlin"


# settings for mode 'local scaling factors'
elif mode == 'local scaling factors':

    filename_output = 'tpamap_tum_mcs.csv'

    # list names of local tire model scaling maps which should be concatenated
    # NOTE: concatenating local acceleration limit data is not intended with this function. This function is build for
    # concatenating local tire scaling factors. This is just an example.
    list_tpamaps = ["tpamap_berlin__27mps.csv",
                    "tpamap_berlin__56mps.csv",
                    "tpamap_berlin__83mps.csv"]

    # list the specific timesteps where each map should be used raw (100%);
    # the maps are interpolated between those timesteps
    time_interpsteps = [0.0, 20.0, 35.0]

    # set to True if map should be used within vehicle dynamics simulation;
    # the file is always necessary, therefore, set to False if no varying friction influence is needed
    bool_enable_tpamaps = True


# Manage paths ---------------------------------------------------------------------------------------------------------

path2tpamaps = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions', 'outputs')

filepath2output = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0],
                               'tpa_map_functions', 'outputs', filename_output)

# concatenate map data -------------------------------------------------------------------------------------------------

if mode == 'local acceleration limits':

    tmf.helperfuncs.concat_tpamaps_ltpl.concat_tpamaps_ltpl(path2tpamaps=path2tpamaps,
                                                            filepath2output=filepath2output,
                                                            tpamap_identifier=tpamap_identifier)

elif mode == 'local scaling factors':

    tmf.helperfuncs.concat_tpamaps_vehdynsim.concat_tpamaps_vehdynsim(path2tpamaps=path2tpamaps,
                                                                      filepath2output=filepath2output,
                                                                      list_tpamaps=list_tpamaps,
                                                                      time_interpsteps=time_interpsteps,
                                                                      bool_enable_tpamaps=bool_enable_tpamaps)
