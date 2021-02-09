import numpy as np
import matplotlib.pyplot as plt
#import tikzplotlib

s_map_m = np.arange(0, 80, 10)
acc_mps2 = np.asarray([11.0, 12.0, 12.5, 13.0, 12.0, 11.0, 10.0, 12.0])
acc_mps2_interp = np.asarray([11.0, 11.0, 12.0, 12.5, 12.0, 11.0, 10.0, 10.0])

s_stepsize_traj = np.arange(0, 80, 5.5)
s_stepsize_traj_shifted = s_stepsize_traj + 2

# acc_result_traj = np.full((1, s_stepsize_traj.size), 10)

xlim = [0, 70]
ylim = [9.5, 13.5]

# calc s coordinate

acc_received_wo_interp = list()
acc_received_with_interp = list()

for s_coordinates in [s_stepsize_traj, s_stepsize_traj_shifted]:

    dummy_acc_rcvd_wo_interp = list()
    dummy_acc_rcvd_with_interp = list()

    for s_m in s_coordinates:

        idx_min = np.argmin(abs(s_map_m - s_m))

        if (s_map_m - s_m)[idx_min] == 0:
            dummy_acc_rcvd_wo_interp.append(acc_mps2[idx_min])
            dummy_acc_rcvd_with_interp.append(acc_mps2_interp[idx_min])

        elif (s_map_m - s_m)[idx_min] > 0:
            dummy_acc_rcvd_wo_interp.append(acc_mps2[idx_min - 1])
            dummy_acc_rcvd_with_interp.append(np.interp(s_m, s_map_m[idx_min - 1:idx_min + 1],
                                                        acc_mps2_interp[idx_min - 1:idx_min + 1]))

        elif (s_map_m - s_m)[idx_min] < 0:
            dummy_acc_rcvd_wo_interp.append(acc_mps2[idx_min])
            dummy_acc_rcvd_with_interp.append(np.interp(s_m, s_map_m[idx_min:idx_min + 2],
                                                        acc_mps2_interp[idx_min:idx_min + 2]))

    acc_received_wo_interp.append(dummy_acc_rcvd_wo_interp)
    acc_received_with_interp.append(dummy_acc_rcvd_with_interp)

# prepare plot filling

x = np.hstack((s_stepsize_traj[:, np.newaxis], s_stepsize_traj[:, np.newaxis],
               s_stepsize_traj_shifted[:, np.newaxis], s_stepsize_traj_shifted[:, np.newaxis]))

x = np.hstack(x)

y1 = acc_received_wo_interp[0][0:2] * 2
y2 = acc_received_wo_interp[1][0:2] * 2

y3 = acc_received_with_interp[0][0:2] * 2
y4 = acc_received_with_interp[1][0:2] * 2

for int_counter, s in enumerate(x[4:-4:2]):

    if np.min(np.abs(s_stepsize_traj - s)) == 0:
        idx_min = np.argmin(np.abs(s_stepsize_traj - s))

        y1.extend(((acc_received_wo_interp[0][idx_min - 1]), (acc_received_wo_interp[0][idx_min])))
        y3.extend(((acc_received_with_interp[0][idx_min - 1]), (acc_received_with_interp[0][idx_min])))

        idx_min = np.argmin(np.abs(s_stepsize_traj_shifted - s))

        if (s_stepsize_traj_shifted[idx_min] - s) > 0:
            y2.extend(((acc_received_wo_interp[1][idx_min - 1]), (acc_received_wo_interp[1][idx_min - 1])))
            y4.extend(((acc_received_with_interp[1][idx_min - 1]), (acc_received_with_interp[1][idx_min - 1])))

        elif (s_stepsize_traj_shifted[idx_min] - s) < 0:
            y2.extend(((acc_received_wo_interp[1][idx_min]), (acc_received_wo_interp[1][idx_min])))
            y4.extend(((acc_received_with_interp[1][idx_min - 1]), (acc_received_with_interp[1][idx_min - 1])))

        else:
            ValueError()

    elif np.min(np.abs(s_stepsize_traj_shifted - s)) == 0:
        idx_min = np.argmin(np.abs(s_stepsize_traj_shifted - s))

        y1.extend(((acc_received_wo_interp[0][idx_min]), (acc_received_wo_interp[0][idx_min])))
        y2.extend(((acc_received_wo_interp[1][idx_min - 1]), (acc_received_wo_interp[1][idx_min])))
        y3.extend(((acc_received_with_interp[0][idx_min]), (acc_received_with_interp[0][idx_min])))
        y4.extend(((acc_received_with_interp[1][idx_min - 1]), (acc_received_with_interp[1][idx_min])))

# y1_tilde = list()

# for ele in np.hstack(np.hstack((s_stepsize_traj[:, np.newaxis], s_stepsize_traj_shifted[:, np.newaxis]))):
#
#     idx_min1 = np.argmin(abs(s_stepsize_traj - ele))
#     idx_min2 = np.argmin(abs(s_stepsize_traj_shifted - ele))
#
#     if (s_stepsize_traj - ele)[idx_min1] == 0:
#         y1_tilde.append(acc_received_wo_interp[0][idx_min1])
#         y1_tilde.append(acc_received_wo_interp[0][idx_min1])
#
#     elif (s_stepsize_traj - ele)[idx_min1] > 0:
#         y1_tilde.append(acc_received_wo_interp[0][idx_min1])
#         y1_tilde.append(acc_received_wo_interp[0][idx_min1])
#
#     elif (s_stepsize_traj - ele)[idx_min1] < 0:
#         y1_tilde.append(acc_received_wo_interp[0][idx_min1])
#         y1_tilde.append(acc_received_wo_interp[0][idx_min1])


# y1 = [11, 11, 11, 11, 11, 11, 11, 11,
#       11, 12, 12, 12, 12, 12, 12, 12,
#       12, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,
#       12.5, 13, 13, 13, 13, 13, 13, 13,
#       13, 12, 12, 12, 12, 12, 12, 12,
#       12, 11, 11, 11,
#       11, 10, 10, 10, 10, 10, 10, 10, 10, 10]
#
# y2 = [11, 11, 11, 11, 11, 11, 11, 11,
#       11, 11, 11, 12, 12, 12, 12, 12,
#       12, 12, 12, 12.5, 12.5, 12.5, 12.5, 12.5,
#       12.5, 12.5, 12.5, 13, 13, 13, 13, 12,
#       12, 12, 12, 12, 12, 12, 12,
#       11, 11, 11, 11,
#       11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10]

# plot figure ----------------------------------------------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)

# upper subplot


ax0.step(s_map_m, acc_mps2, where='post', color='k', linewidth=2.0, label='stored values')

ax0.scatter(s_stepsize_traj, acc_received_wo_interp[0], marker='x', label='path coordinates at t_0')
ax0.step(s_stepsize_traj, acc_received_wo_interp[0], where='post', label='acc. limits at t_0')
ax0.scatter(s_stepsize_traj_shifted, acc_received_wo_interp[1], marker='o', label='path coordinates at t_1')
ax0.step(s_stepsize_traj_shifted, acc_received_wo_interp[1], where='post', label='acc. limits at t_1')


# ax0.scatter(x,y1, marker='+')
# ax0.scatter(x,y2, marker='*')
ax0.fill_between(x[0:len(y1)], y1, y2, where=(y1 > y2), alpha=0.2, color='k', hatch='/')


# lower subplot

ax1.step(s_map_m, acc_mps2, where='post', color='k', linewidth=2.0, label='stored values')
ax1.plot(s_map_m, acc_mps2_interp, color='k', linewidth=2.0, linestyle='--', label='interpolated values')

ax1.scatter(s_stepsize_traj, acc_received_with_interp[0], marker='x', label='path coordinates at t_0')
ax1.step(s_stepsize_traj, acc_received_with_interp[0], where='post', label='acc. limits at t_0')
ax1.scatter(s_stepsize_traj_shifted, acc_received_with_interp[1], marker='o', label='path coordinates at t_1')
ax1.step(s_stepsize_traj_shifted, acc_received_with_interp[1], where='post', label='acc. limits at t_1')

ax1.fill_between(x[0:len(y3)], y3, y4, where=(y3 > y4), alpha=0.2, color='k', hatch='/')


# configure axis
ax0.set_xlabel('s in meters')
ax0.set_ylabel('acceleration limits in m/s^2')
ax1.set_xlabel('s in meters')
ax1.set_ylabel('acceleration limits in m/s^2')

ax0.set_xlim(xlim)
ax0.set_ylim(ylim)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)

ax0.legend()
ax1.legend()

# export to tikz
plt.draw()
#tikzplotlib.save('discretization_frictionmap.tex')

plt.show()
