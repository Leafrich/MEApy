import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from itertools import chain


def in_range(pd_data):
    burst_in_range = []
    for k in range(len(pd_data['Channel Label'])):
        if range_in_s[1] >= pd_data['Start timestamp [µs]'].iloc[k] / 1e6 >= range_in_s[0]:  # start within range
            if pd_data['End timestamp [µs]'].iloc[k] / 1e6 < range_in_s[1]:  # end within range
                burst_in_range.append([pd_data['Start timestamp [µs]'].iloc[k],
                                       pd_data['End timestamp [µs]'].iloc[k]])
            else:  # start in range/ end not in range
                burst_in_range.append([pd_data['Start timestamp [µs]'].iloc[k],
                                       timeStop * 1e6])
                break
        else:  # start not in range
            if range_in_s[1] >= pd_data['End timestamp [µs]'].iloc[k] / 1e6 >= range_in_s[0]:  # end in range
                burst_in_range.append([timeStart * 1e6, pd_data['End timestamp [µs]'].iloc[k]])
    return burst_in_range


def sep_bursts():
    array = []
    bur_int = [j for sub in burst_in_range for j in sub]
    bur_int = np.diff(bur_int)
    sep_bur = []
    slBurst = []

    for n in range(len(bur_int)):
        if n % 2 != 0:
            slBurst.append(bur_int[n])

    for m in range(int(len(slBurst))):
        array.append(burst_in_range[m])
        if slBurst[m] > 500000:
            sep_bur.append(array)
            array = []
    burst_array.append(sep_bur)


pd.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)

timeStart = 0
timeStop = 100
range_in_s = (timeStart, timeStop)

rootDir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f'{rootDir}\\Hippocampus bursts analysis.csv')
df = df.iloc[:, [1, 3, 6, 9, 10, 11, 12]]
df['End timestamp [µs]'] = df['Start timestamp [µs]'] + df['Duration [µs]']

uniNames = pd.unique(df['Experiment'])
DIVarr = ['DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24']  # in future experiments this could become uni val "experiment"
Dict = {}

df['ID'] = df['Well Label'].astype(str) + df['Channel Label'].astype(str)  # not needed, remove in time when replaced

for _ in range(len(uniNames)):
    Dict[DIVarr[_]] = df.loc[df['Experiment'] == uniNames[_], :]

burst_array = []

for i in DIVarr:  # DIV seperationg
    burst_array = []

    # need another loop to loop through the wells first then IDs
    # ID column is not needed
    for _ in pd.unique(Dict[str(i)]["ID"]):  # ID seperation
        j = df.loc[df['ID'] == _, :]
        burst_in_range = in_range(j)

        x = np.asarray(list(chain.from_iterable(burst_in_range)))
        plt.scatter(x / 1e6, [_] * len(x), color='#FFAF4A', marker='s', zorder=2, s=2)

        # Sep burst function needs to be finished


    plt.show()
