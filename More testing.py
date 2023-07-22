import matplotlib.pyplot as plt
import numpy as np

import os
import McsPy.McsData
from McsPy import ureg, Q_
import matplotlib.pyplot as plt

# Some example data to display
# x = np.linspace(0, 2 * np.pi, 400)
# y = np.arange(0, 400, 1)
# z = np.sin(x ** 2)
#
# font = {'size': 5}
# plt.rc('font', **font)
#
# MCS24Dict = {
#     "0": 0,
#     "1": z,
#     "2": z,
#     "3": 0,
#     "4": z,
#     "5": z,
#     "6": z,
#     "7": z,
#     "8": -z,
#     "9": -z,
#     "10": -z,
#     "11": -z,
#     "12": 0,
#     "13": -z,
#     "14": -z,
#     "15": 0,
# }
#
# fig, pos = plt.subplots(4, 4)
# pos[0, 0].axis('off')
# pos[0, 3].axis('off')
# pos[3, 0].axis('off')
# pos[3, 3].axis('off')
# # pos[0, 1].set_title('Axis [0, 1]')
# # pos[1, 1].plot(x, -y, 'tab:red')
#
# n = 0
# for i in range(4):
#     for j in range(4):
#         if n == 0 or n == 3 or n == 12 or n == 15:
#             n += 1
#             continue
#         print(i, j)
#         pos[i, j].plot(y, MCS24Dict.get(str(n)))
#         n += 1
#
# # for ax in pos.flat:
# #     ax.set(xlabel='x-label', ylabel='y-label')
#
# # Hide x labels and tick labels for top plots and y ticks for right plots.
# # for ax in pos.flat:
# #     ax.label_outer()
#
# plt.show()

