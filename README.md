# Description
This repository provides several functions to process the local acceleration limitations for trajectory planning at
TUM/FTM. In addition to location-dependent acceleration limits, vehicle velocity dependent acceleration limits can be
accessed when provided.
The initial acceleration limits are loaded from a csv-file. There is an option to update the local acceleration in
real-time when the tire performance assessment (tpa) module is running in parallel and communication is enabled.

The picture below shows an example tire performance map of the Berlin Formula-e race track.

![tpamap_berlin](/resources/tpamap_berlin.png)

# List of components
* `tpa_map_helperfuncs`: This folder contains some helper functions used within this repository.
* `tpa_map_interface`: This folder contains functions to provide an interface to the local trajectory planner and to the
tire performance assessment module.
* `tpa_map_visualization`: This folder contains functions to visualize tpa maps.
