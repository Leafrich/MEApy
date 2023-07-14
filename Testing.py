import os
import McsPy.McsData
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
import numpy as np


# MCS24Dict = {
#     "0": 0, "1": z, "2": z, "3": 0,
#     "4": z, "5": z, "6": z, "7": z,
#     "8": z, "9": -z, "10": -z, "11": -z,
#     "12": 0, "13": -z, "14": -z, "15": 0,
# }

def plot_well(analog_streams, channel_idx, from_in_s=0, to_in_s=None, show=False):
    """
    Plots data from a single AnalogStream channel

    :param analog_streams: A AnalogStream array
    :param channel_idx: A scalar channel index (0 <= channel_idx < # channels in the AnalogStream)
    :param from_in_s: The start timestamp of the plot (0 <= from_in_s < to_in_s). Default: 0
    :param to_in_s: The end timestamp of the plot (from_in_s < to_in_s <= duration). Default: None (= recording duration)
    :param show: If True (default), the plot is directly created. For further plotting, use show=False
    """
    # extract basic information
    ids = [c.channel_id for c in analog_streams.channel_infos.values()]
    channel_id = ids[channel_idx]
    channel_info = analog_streams.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude

    # get start and end index
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_streams.channel_data.shape[1]
    else:
        to_idx = min(analog_streams.channel_data.shape[1], int(to_in_s * sampling_frequency))

    # get the timestamps for each sample
    time = analog_streams.get_channel_sample_timestamps(channel_id, from_idx, to_idx)

    # scale time to seconds:
    scale_factor_for_second = Q_(1, time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second

    # scale signal to µV:
    scale_signal = analog_streams.get_channel_in_range(channel_id, from_idx, to_idx)
    scale_factor_for_uV = Q_(1, scale_signal[1]).to(ureg.uV).magnitude

    # construct the plot
    fig, pos = plt.subplots(4, 4)
    pos[0, 0].axis('off')
    pos[0, 3].axis('off')
    pos[3, 0].axis('off')
    pos[3, 3].axis('off')

    n = 0
    for i in range(4):
        for j in range(4):
            if n == 0 or n == 3 or n == 12 or n == 15:
                n += 1
                continue
            print(n)
            signal = analog_streams.get_channel_in_range(n, from_idx, to_idx)
            pos[i, j].plot(time_in_sec, signal * scale_factor_for_uV)
            n += 1

    _ = plt.xlabel('Time (%s)' % ureg.s)
    _ = plt.ylabel('Voltage (%s)' % ureg.uV)
    _ = plt.title('Well XX')
    if show:
        plt.show()


well_id = 1
electrode_id = 6
timeStart = 0
timeStop = 300

rootDir = os.path.dirname(os.path.abspath(__file__))
D5data = f'{rootDir}\\20230530_Cortex_pMEA.h5'
file = McsPy.McsData.RawData(D5data)
electrode_stream = file.recordings[0].analog_streams[0]
ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
channel_id = ids[electrode_id]

info = electrode_stream.channel_infos[channel_id].info

print("-----------------------------------------------------------")
print("Sampling frequency : %s Hz" % electrode_stream.channel_infos[channel_id].sampling_frequency.magnitude)
print("Bandwidth : %s - %s Hz" % (info['HighPassFilterCutOffFrequency'], info['LowPassFilterCutOffFrequency']))
print("-----------------------------------------------------------")

signal = electrode_stream.get_channel_in_range(channel_id, 0, electrode_stream.channel_data.shape[1])[0]

plot_well(electrode_stream, 1, from_in_s=timeStart, to_in_s=timeStop, show=False)
