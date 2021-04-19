import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt

# import custom modules
path2tmf = os.path.join(os.path.abspath(__file__).split('tpa_map_functions')[0], 'tpa_map_functions')
sys.path.append(path2tmf)

import tpa_map_functions as tmf

# User Input -------------------------------------------------------------------------------------------------------

track_name = 'berlin'
bool_enable_debug = True

mode_resample_refline = 'var_steps'
stepsize_resample_m = 11.11
section_length_min_m = 15
section_length_max_m = 200

test_source = 'path'  # or 'path'

# Preprocess Reference Line ----------------------------------------------------------------------------------------

filepath2ltpl_refline = os.path.join(path2tmf, 'inputs', 'traj_ltpl_cl', 'traj_ltpl_cl_' + track_name + '.csv')

if test_source == 'file':

    # load reference line
    with open(filepath2ltpl_refline, 'r') as fh:
        csv_data_refline = np.genfromtxt(fh, delimiter=';')

    reference_line = csv_data_refline[:, 0:2]

    mode_resample_refline = 'const_steps'

    output_data = tmf.helperfuncs.preprocess_ltplrefline.preprocess_ltplrefline(reference_line=reference_line,
                                                                                stepsize_resample_m=stepsize_resample_m,
                                                                                bool_enable_debug=bool_enable_debug)
else:

    output_data = tmf.helperfuncs.preprocess_ltplrefline.\
        preprocess_ltplrefline(filepath2ltpl_refline=filepath2ltpl_refline,
                               mode_resample_refline=mode_resample_refline,
                               stepsize_resample_m=stepsize_resample_m,
                               section_length_limits_m=[section_length_min_m, section_length_max_m],
                               bool_enable_debug=bool_enable_debug)

test = np.concatenate((output_data['refline_resampled']['section_id'][:, np.newaxis],
                       output_data['refline_resampled']['refline_resampled']), axis=1)

if bool_enable_debug:

    refline_original = output_data['refline']
    refline_resampled = output_data['refline_resampled']['refline_resampled']

    if mode_resample_refline == "const_steps":

        plt.figure(figsize=(7, 7))

        plt.plot(refline_original[:, 1], refline_original[:, 2], 'k--', label='original reference line')
        plt.plot(refline_original[:, 1], refline_original[:, 2], 'kx', label='original reference line')
        plt.plot(refline_resampled[:, 1], refline_resampled[:, 2], 'r', label='resampled reference line')
        plt.plot(refline_resampled[:, 1], refline_resampled[:, 2], 'ro', label='resampled reference line')

        plt.axis('equal')
        plt.legend()
        plt.xlabel('x in meters')
        plt.ylabel('y in meters')

        plt.show(block=False)

        # plot histogram containing distances between coordinate points
        plt.figure()

        plt.hist(np.sqrt(np.sum(np.diff(refline_resampled[:, 1:3], axis=0) ** 2, axis=1)), bins=20)

        plt.axvline(x=output_data['refline_resampled']['mean_diff_m'], color='g', label='mean')
        plt.axvline(x=(output_data['refline_resampled']['mean_diff_m']
                    + output_data['refline_resampled']['std_diff_m']), color='y', label='stand.dev.')
        plt.axvline(x=(output_data['refline_resampled']['mean_diff_m']
                    - output_data['refline_resampled']['std_diff_m']), color='y')
        plt.axvline(x=output_data['refline_resampled']['min_diff_m'], color='r', label='min/max')
        plt.axvline(x=output_data['refline_resampled']['max_diff_m'], color='r')

        plt.legend()
        plt.grid()
        plt.xlabel('distance between reference line coordinate points in meters')
        plt.ylabel('bin count')

        plt.show()

    elif mode_resample_refline == "var_steps":

        idxs_plot = output_data['refline_resampled']['sectionid_change']

        plt.figure(figsize=(7, 7))

        plt.plot(refline_original[:, 1], refline_original[:, 2], 'k--', label='original reference line')
        plt.plot(refline_original[:, 1], refline_original[:, 2], 'kx', label='original reference line')
        plt.plot(refline_resampled[:, 1][idxs_plot], refline_resampled[:, 2][idxs_plot], 'r',
                 label='resampled reference line')
        plt.plot(refline_resampled[:, 1][idxs_plot], refline_resampled[:, 2][idxs_plot], 'ro',
                 label='resampled reference line')

        plt.axis('equal')
        plt.legend()
        plt.xlabel('x in meters')
        plt.ylabel('y in meters')

        plt.show(block=False)

        plt.figure()

        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(refline_original[:, 0], output_data['refline_resampled']['ax_mps2'], label="long. acc.")
        ax1.plot(refline_original[:, 0], output_data['refline_resampled']['ay_mps2'], label="lat. acc.")

        for s in refline_resampled[:, 0][idxs_plot]:
            plt.vlines(s, -10, 10, colors='k', linestyle='--')

        plt.grid()
        plt.xlabel("track position in m")
        plt.ylabel("long./lat. acc. in mps2")
        plt.legend()

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)

        ax2.step(refline_resampled[:, 0][idxs_plot],
                 np.multiply(output_data['refline_resampled']['ax_trigger'][idxs_plot], 0.9),
                 where='post', linewidth=2.0, label="trigger: long. acc.")
        ax2.step(refline_resampled[:, 0][idxs_plot],
                 np.multiply(output_data['refline_resampled']['ay_trigger'][idxs_plot], 0.8),
                 where='post', linewidth=2.0, label="trigger: lat. acc.")

        ax2.step(refline_resampled[:, 0],
                 np.multiply(output_data['refline_resampled']['list_section_category'], 1.0), where='post',
                 linewidth=2.0, label="section type")

        for s in refline_resampled[:, 0][idxs_plot]:
            plt.vlines(s, -7, 7, colors='k', linestyle='--')

        plt.ylim([-7, 7])

        plt.grid()
        plt.xlabel("track position in m")
        plt.ylabel("section type")

        plt.legend()
        plt.show()
