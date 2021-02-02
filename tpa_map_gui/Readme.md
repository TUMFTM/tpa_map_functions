# Generation of tpa maps via GUI

## Overview
This Readme explains how to create tpa maps which provide variable friction data
for simulation and acceleration limits for trajectory planning.

## Setup
1. Create and activate a virtual environment.
2. Install the requirements listed in `requirements.txt`.

## How to gernerate tpa maps
1. Open the main script `main_tpaGUI.py` and adjust the variables within the section `User Input`. Documentation is provided within the script.
2. Run the python script `main_tpaGUI.py` to open the GUI.
    * Insert desired local scaling factors (gui_mode = 1) or long./lat. acceleration values (gui_mode = 2) or set to random or smooth.
    * Optionally insert an appendix which is added to the name of the tpa map
    that you want to generate.
    * Click `save map` to store the tpa map into the `/outputs` folder.
3. Repeat step 2 until all desired tpa maps are generated.

The next steps will create `tpamap_tum_mcs.csv` which holds all information on a variable friction scenario that is processed by simulations in mod_control or sim_vehicle_dynamics.

4. Open the file `tpa_map_functions/helperfuncs/tpamap_concatenate.py`
    * Insert the names of the previously created tpa maps which should be part of a variable friction scenario simulated later in a mod_control or sim_vehicle_dynamics simulation (max 10 maps).
    * Enter the simulation times in the `time_interpsteps` array which are related to 100% activation of the corresponding tpa map (eg. `[0 10 20]` if map2 is 100% active at 10s and map3 100% active at 20s).
    * Set `bool_enable_tpamaps` to `True`
5. Run the python script `tpamap_concatenate.py`.

The generated `tpamap_tum_mcs.csv` file is located in `/outputs`.

## tpa map structure
The generated tpa map `tpamap_tum_mcs.csv` has the size [2502 x 23].
* `bool_enable_tpamaps` is a scalar located in row 1 & column 1
* `time_interpsteps` is located in row 2 and column 1 to 10
* the actual tpa map, size [2500 x 23], is saved in row 2 to 2502 and column 1 to 23. Column 1 holds s_m, column 2 holds x_m, column 3 holds y_m. Column 4 upwards holds the local scaling factors from the concatenated tpa maps.

## Process pipeline
Further information on how to integrate the generated tpa map is provided in `sim_vehicle_dynamics\vehicle_environment\variable_friction\Readme.md`.

Author: [Dominik Staerk](mailto:dominik.staerk@tum.de)

Contact person: [Leonhard Hermansdorfer](mailto:leo.hermansdorfer@tum.de)
