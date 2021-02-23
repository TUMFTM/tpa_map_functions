# tpa_map_functions

The software of this repository is used within the [TUM Autonomous Motorsports project](https://www.mw.tum.de/en/ftm/main-research/intelligent-vehicle-systems/roborace-autonomous-motorsport/).

The core functionality of this repository (in ``/tpa_map_functions``) is available as pypi package: https://pypi.org/project/tpa-map-functions/

## Description
This repository provides several functions to generate and process race track maps containing specific local information, which is further used for trajectory planning or vehicle dynamics simulation.

### local acceleration limits
The local acceleration limits are used for trajectory planning within the TUM Autonomous Motorsports project. These location-dependent values are stored in a tpa-map (tpa = tire performance assessment) and are provided to the local trajectory planning module via an interface. In addition to location-dependent acceleration limits, vehicle velocity-dependent acceleration limits can be included (e.g. to account for aerodynamic forces acting on the vehicle (downforce)).

These tpa-maps are static, but there is an option to update the local acceleration limits in real-time when the tire performance assessment (tpa) module is running in parallel and communication is enabled. This allows real-time adaption to changing race track conditions.

### local tire model scaling
The local tire model scaling is used for vehicle dynamics simulation within the TUM Autonomous Motorsports project. These maps have the same format but contain local scaling factors for the tire model. This allows simulate a varying tire-road friction coefficient. The local tire model scaling maps allow location-dependent and time-dependent scaling factors.

The picture below shows an example tire performance map of the Berlin Formula-e race track.

![tpamap_berlin](/resources/tpamap_berlin.png)

## List of components
* `inputs`: This folder contains input files (e.g. reference line or tpa-maps).
* `outputs`: This folder contains generated tpa-maps.
* `tests`: This folder contains scripts to test several functions within this folder (e.g. interface testing).
* `tpa_map_functions/helperfuncs`: This folder contains some helper functions used within this repository.
* `tpa_map_functions/interface`: This folder contains functions to provide an interface to the local trajectory planner and to the
tire performance assessment module.
* `tpa_map_functions/visualization`: This folder contains functions to visualize tpa-maps.
* `tpa_map_gui`: This folder contains software to run a GUI which allows to generate location- and time dependent tire-road friction maps for vehicle dynamics simulation or to generate tpa-maps which are used for trajectory planning. A more detailed description can be found within this folder.

## How to generate tpa-maps
With these steps, a new tpa-map can be generated using an existing reference line:
1. Open ``main_tpaGUI.py`` and specify the name of the reference line (in ``/inputs``) and the settings for ref-line preprocessing (options: use original step size or resample to new step size).
2. Create a tpa-map (consult ``/tpa_map_gui/Readme.md`` for more details).
3. Reformat output maps depending on type (local scaling map -> ``/tpa_map_functions/helperfuncs/concat_tpamaps_vehdynsim.py``; only for multiple tpa-maps with velocity-dependent data -> ``/tpa_map_functions/helperfuncs/concat_tpamaps_lptl.py``).
4. Use final, single map file (located in ``/outputs``) as input for local trajectory planner or vehicle dynamics simulation.
