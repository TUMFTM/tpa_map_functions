# Description
This repository provides several functions to process the local acceleration limitations for trajectory planning at
TUM/FTM. In addition to location-dependent acceleration limits, vehicle velocity dependent acceleration limits can be
accessed when provided.
The initial acceleration limits are loaded from a csv-file. There is an option to update the local acceleration in
real-time when the tire performance assessment (tpa) module is running in parallel and communication is enabled.

The picture below shows an example tire performance map of the Berlin Formula-e race track.

![tpamap_berlin](/resources/tpamap_berlin.png)

# List of components
* `inputs`: This folder contains input files (e.g. reference line or tpa maps).
* `outputs`: This folder contains generated tpa maps.
* `tests`: This folder contains scripts to test several functions within this folder (e.g. interface testing).
* `tpa_map_functions/helperfuncs`: This folder contains some helper functions used within this repository.
* `tpa_map_functions/interface`: This folder contains functions to provide an interface to the local trajectory planner and to the
tire performance assessment module.
* `tpa_map_functions/visualization`: This folder contains functions to visualize tpa maps.
* `tpa_map_gui`: This folder contains software to run a GUI which allows to generate location- and time dependent tire-road friction maps for vehicle dynamics simulation or to generate tpa maps which are used for trajectory planning. A more detailed description can be found within this folder.
