import os
import McsPy.McsData
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_analog_stream_channel(analog_stream, channel_idx, from_in_s=0, to_in_s=None, show=False):
    """
    Plots data from a single AnalogStream channel

    :param analog_stream: A AnalogStream object
    :param channel_idx: A scalar channel index (0 <= channel_idx < # channels in the AnalogStream)
    :param from_in_s: The start timestamp of the plot (0 <= from_in_s < to_in_s). Default: 0
    :param to_in_s: The end timestamp of the plot (from_in_s < to_in_s <= duration).
    :param show: If True (default), the plot is directly created. For further plotting, use show=False
    """
    # extract basic information
    ids = [c.channel_id for c in analog_stream.channel_infos.values()]
    channel_id = ids[channel_idx]
    channel_info = analog_stream.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude

    # get start and end index
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_stream.channel_data.shape[1]
    else:
        to_idx = min(analog_stream.channel_data.shape[1], int(to_in_s * sampling_frequency))

    # get the timestamps for each sample
    time = analog_stream.get_channel_sample_timestamps(channel_id, from_idx, to_idx)

    # scale time to seconds:
    scale_factor_for_second = Q_(1, time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second

    # get the signal
    signal = analog_stream.get_channel_in_range(channel_id, from_idx, to_idx)

    # scale signal to µV:
    scale_factor_for_uV = Q_(1, signal[1]).to(ureg.uV).magnitude
    signal_in_uV = signal[0] * scale_factor_for_uV

    # construct the plot
    _ = plt.figure(figsize=(16, 5))
    _ = plt.plot(time_in_sec, signal_in_uV, color='#94D9A2', linewidth=0.2)
    _ = plt.xlabel('Time (%s)' % ureg.s)
    _ = plt.ylabel('Voltage (%s)' % ureg.uV)
    _ = plt.title('Channel %s' % channel_info.info['Label'])
    if show:
        plt.show()


rootDir = os.path.dirname(os.path.abspath(__file__))
D5data = f'{rootDir}\\wD6_D18_hc.h5'
data = McsPy.McsData.RawData(D5data)
stream = data.recordings[0].analog_streams[0]
ids = [c.channel_id for c in stream.channel_infos.values()]  # needed to find proper channel_data rows for .get_channel
info = stream.channel_infos[ids[0]].info

timeStart = 0
timeStop = 300

# Methods extraction, .channel_infos not found?
# _ = [method for method in dir(stream) if callable(getattr(stream, method))]
# for i in range(len(_)):
#     print(_[i])

# Finding info columns works well
# well_ids = [c.group_id for c in stream.channel_infos.values()]
# print(well_ids)

data_array = []
for x in range(len(ids)):
    data_array.append([y for y in stream.get_channel(ids[x])[0]])

print("-----------------------------------------------------------")
print("Sampling frequency : %s Hz" % stream.channel_infos[ids[0]].sampling_frequency.magnitude)
print("Bandwidth : %s - %s Hz" % (info['HighPassFilterCutOffFrequency'], info['LowPassFilterCutOffFrequency']))
print("-----------------------------------------------------------")

fs = int(stream.channel_infos[ids[3]].sampling_frequency.magnitude)

df = pd.read_csv("../Burst analysis.csv")
df = df.iloc[:, [1, 3, 9, 10, 11]]
df['End timestamp [µs]'] = df['Start timestamp [µs]'] + df['Duration [µs]']

# Non bursting channels do NOT show up!
# Keys need to be actual IDs, no reason to use a dict otherwise
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
    _ = plt.scatter(burst_dot, [-1] * 200, color='#FB4629', marker='s', s=2, zorder=3)

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
    _ = plt.scatter(sep_burst_dot, [1] * 500, color='#FFAF4A', marker='s', zorder=2, s=2)

print(sep_bur[0])

plt.show()
