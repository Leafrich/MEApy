import os
import McsPy.McsData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Functions import plot_analog_stream_channel

rootDir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
D5data = f'{rootDir}\\Data\\wD6_D18_hc.h5'
data = McsPy.McsData.RawData(D5data)
stream = data.recordings[0].analog_streams[0]
ids = [c.channel_id for c in stream.channel_infos.values()]
info = stream.channel_infos[ids[0]].info

timeStart = 0
timeStop = 300

data_array = []
for x in range(len(ids)):
    data_array.append([y for y in stream.get_channel(ids[x])[0]])

print("-----------------------------------------------------------")
print("Sampling frequency : %s Hz" % stream.channel_infos[ids[0]].sampling_frequency.magnitude)
print("Bandwidth : %s - %s Hz" % (info['HighPassFilterCutOffFrequency'], info['LowPassFilterCutOffFrequency']))
print("-----------------------------------------------------------")

fs = int(stream.channel_infos[ids[3]].sampling_frequency.magnitude)

df = pd.read_csv("../Data/Burst data wD6.csv")
df = df.iloc[:, [1, 3, 9, 10, 11]]
df['End timestamp [µs]'] = df['Start timestamp [µs]'] + df['Duration [µs]']

df['ID'] = df['Well Label'].astype(str) + df['Channel Label'].astype(str)
uIDs = df['ID'].unique()
burst_int = np.empty(len(df) - 1)
for i in range(len(df) - 1):
    burst_int[i] = df.loc[i + 1, 'Start timestamp [µs]'] - df.loc[i, 'End timestamp [µs]']
burst_dict = {}
for i in range(len(uIDs)):
    burst_dict[i] = df.loc[df['ID'] == uIDs[i], :]

# Spikes in range
range_in_s = (timeStart, timeStop)
# ts_spks = spks / fs
# spks_in_range = ts_spks[(ts_spks >= range_in_s[0]) & (ts_spks <= range_in_s[1])]

plot_analog_stream_channel(stream, 7, from_in_s=timeStart, to_in_s=timeStop, show=False)
# _ = plt.plot(spks_in_range, [falling_threshold * 1e6] * spks_in_range.shape[0], 'bo', ms=0.4, alpha=0.5, zorder=1)

burst_in_range = []
ID = 5
for i in range(len(burst_dict[ID]['Start timestamp [µs]'])):
    if range_in_s[1] >= burst_dict[ID]['Start timestamp [µs]'].iloc[i] / 1e6 >= range_in_s[0]:  # start within range
        if burst_dict[ID]['End timestamp [µs]'].iloc[i] / 1e6 < range_in_s[1]:  # end within range
            burst_in_range.append([burst_dict[ID]['Start timestamp [µs]'].iloc[i],
                                   burst_dict[ID]['End timestamp [µs]'].iloc[i]])
        else:  # start in range/ end not in range
            burst_in_range.append([burst_dict[ID]['Start timestamp [µs]'].iloc[i],
                                   timeStop * 1e6])
            print('Burst period in range end')
            break
    else:  # start not in range
        if range_in_s[1] >= burst_dict[ID]['End timestamp [µs]'].iloc[i] / 1e6 >= range_in_s[0]:  # end in range
            burst_in_range.append([timeStart * 1e6, burst_dict[ID]['End timestamp [µs]'].iloc[i]])
            print('Burst period in range start')

for i in range(len(burst_in_range)):
    burst_dot = np.linspace(burst_in_range[i][0] / 1e6,
                            burst_in_range[i][1] / 1e6, 200)
    _ = plt.scatter(burst_dot, [0] * 200, color='#7878f0', marker='s', s=2, zorder=3)

array = []
sep_bur = []
bur_int = [j for sub in burst_in_range for j in sub]
bur_int = np.diff(bur_int)
j = 0
slBurst = []

for i in range(len(bur_int)):
    if i % 2 != 0:
        slBurst.append(bur_int[i])

for i in range(int(len(slBurst))):
    array.append(burst_in_range[i])
    if slBurst[i] > 500000:
        sep_bur.append(array)
        array = []

for i in range(len(sep_bur)):
    sep_burst_dot = np.linspace(sep_bur[i][0][0] / 1e6,
                                sep_bur[i][-1][1] / 1e6, 500)
    _ = plt.scatter(sep_burst_dot, [5] * 500, color='#4a4ad4', marker='s', zorder=2, s=2)

plt.show()
