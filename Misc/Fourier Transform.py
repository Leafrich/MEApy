import os
import McsPy.McsData
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq, irfft

# MCS24Dict = {
#     "0": 0, "1": z, "2": z, "3": 0,
#     "4": z, "5": z, "6": z, "7": z,
#     "8": z, "9": z, "10": z, "11": z,
#     "12": 0, "13": z, "14": z, "15": 0,
# }

def plot_well(analog_streams, from_in_s=0, to_in_s=None, show=False):
    """
    Plots data from a single AnalogStream channel

    :param analog_streams: Data stream
    :param from_in_s: The start timestamp of the plot (0 <= from_in_s < to_in_s). Default: 0
    :param to_in_s: The end timestamp of the plot (from_in_s < to_in_s <= duration). Default: None (= recording duration)
    :param show: If True (default), the plot is directly created. For further plotting, use show=False
    """
    # extract basic information
    ids = [c.channel_id for c in analog_streams.channel_infos.values()]
    channel_id = ids[0]
    channel_info = analog_streams.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude

    # get start and end index
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_streams.channel_data.shape[1]
    else:
        to_idx = min(analog_streams.channel_data.shape[1], int(to_in_s * sampling_frequency))

    # Messy way of finding out the length of the array
    lenSig = analog_streams.get_channel_in_range(ids[0], from_idx, to_idx)

    # get the timestamps for each sample
    time = analog_streams.get_channel_sample_timestamps(channel_id, from_idx, to_idx)

    # scale time to seconds:
    scale_factor_for_second = Q_(1, time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second

    # scale signal to µV:
    scale_signal = analog_streams.get_channel_in_range(channel_id, from_idx, to_idx)
    scale_factor_for_uV = Q_(1, scale_signal[1]).to(ureg.uV).magnitude

    # remove corner plots
    fig, pos = plt.subplots(4, 4)
    pos[0, 0].axis('off')
    pos[0, 3].axis('off')
    pos[3, 0].axis('off')
    pos[3, 3].axis('off')

    sigArray = np.empty(shape=(12, len(lenSig[0])))
    for j in range(12):
        signal = analog_streams.get_channel_in_range(ids[j], from_idx, to_idx)
        sigArray[j] = signal[0]

    n = 0
    sigDict = {}
    keys = range(16)
    for i in keys:
        if i == 0 or i == 3 or i == 12 or i == 15:
            sigDict[i] = 0
            continue
        sigDict[i] = sigArray[n]
        n += 1

    n = 0
    for i in range(4):
        for j in range(4):
            if n == 0 or n == 3 or n == 12 or n == 15:
                n += 1
                continue
            signal = sigDict[n]
            pos[i, j].title.set_text(n)
            pos[i, j].plot(time_in_sec, signal * scale_factor_for_uV, color='#9BD8BA', linewidth=0.1)
            n += 1

    # _ = plt.figure(figsize=(15, 5)) # Only applies to one subplot
    fig.set_figheight(8)
    fig.set_figwidth(12)

    # Only applies to one subplot
    # _ = plt.xlabel('Time (%s)' % ureg.s)
    # _ = plt.ylabel('Voltage (%s)' % ureg.uV)
    if show:
        plt.show()


def plot_well_fft(analog_streams, from_in_s=0, to_in_s=None, show=False):
    """
    Plots data from a single AnalogStream channel

    :param analog_streams: Data stream
    :param from_in_s: The start timestamp of the plot (0 <= from_in_s < to_in_s). Default: 0
    :param to_in_s: The end timestamp of the plot (from_in_s < to_in_s <= duration). Default: None (= recording duration)
    :param show: If True (default), the plot is directly created. For further plotting, use show=False
    """
    # extract basic information
    ids = [c.channel_id for c in analog_streams.channel_infos.values()]
    channel_id = ids[0]
    channel_info = analog_streams.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude

    # get start and end index
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_streams.channel_data.shape[1]
    else:
        to_idx = min(analog_streams.channel_data.shape[1], int(to_in_s * sampling_frequency))

    # Messy way of finding out the length of the array
    lenSig = analog_streams.get_channel_in_range(ids[0], from_idx, to_idx)

    # get the timestamps for each sample
    time = analog_streams.get_channel_sample_timestamps(channel_id, from_idx, to_idx)

    # scale time to seconds:
    scale_factor_for_second = Q_(1, time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second

    # scale signal to µV:
    scale_signal = analog_streams.get_channel_in_range(channel_id, from_idx, to_idx)
    scale_factor_for_uV = Q_(1, scale_signal[1]).to(ureg.uV).magnitude

    # # remove corner plots
    # fig, pos = plt.subplots(4, 4)
    # pos[0, 0].axis('off')
    # pos[0, 3].axis('off')
    # pos[3, 0].axis('off')
    # pos[3, 3].axis('off')

    sigArray = np.empty(shape=(12, len(lenSig[0])))
    for j in range(12):
        signal = analog_streams.get_channel_in_range(ids[j], from_idx, to_idx)
        sigArray[j] = signal[0]

    Sfreq = 10000
    N = Sfreq * (timeStop - timeStart)
    xf = rfftfreq(N, 1 / Sfreq)

    n = 0
    sigDict = {}
    nsigDict = {}
    keys = range(16)
    for i in keys:
        if i == 0 or i == 3 or i == 12 or i == 15:
            sigDict[i] = 0
            nsigDict[i] = 0
            continue
        sigDict[i] = rfft(sigArray[n])
        nsigDict[i] = sigArray[n]
        n += 1

    n = 0
    for i in range(4):
        for j in range(4):
            if n == 0 or n == 3 or n == 12 or n == 15:
                n += 1
                continue
            signal = sigDict[n]
            # pos[i, j].title.set_text(n)
            # pos[i, j].plot(xf, signal, color='#9BD8BA', linewidth=0.1, label=n, alpha=0.8)
            n += 1

    # fig.set_figheight(8)
    # fig.set_figwidth(12)

    # Only applies to one subplot
    # _ = plt.xlabel('Time (%s)' % ureg.s)
    # _ = plt.ylabel('Voltage (%s)' % ureg.uV)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    return sigDict, nsigDict, xf  # fourier variables


well_id = 1
electrode_id = 6
timeStart = 0
timeStop = 30

rootDir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
D5data = f'{rootDir}\\Data\\wD6_D18_hc.h5'
file = McsPy.McsData.RawData(D5data)
electrode_stream = file.recordings[0].analog_streams[0]
ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
channel_id = ids[electrode_id]

info = electrode_stream.channel_infos[channel_id].info

print("-----------------------------------------------------------")
print("Sampling frequency : %s Hz" % int(electrode_stream.channel_infos[channel_id].sampling_frequency.magnitude))
print("Bandwidth : %s - %s Hz" % (info['HighPassFilterCutOffFrequency'], info['LowPassFilterCutOffFrequency']))
print("-----------------------------------------------------------")

sigDict, nsigDict, xf = plot_well_fft(electrode_stream, from_in_s=timeStart, to_in_s=timeStop, show=True)
yf = sigDict[9]
signal = nsigDict[9]

sr = 10000
points_per_freq = len(xf) / (sr / 2)
target_idx = int(points_per_freq * 1500)
yf[target_idx : len(yf)] = 0
yf[0 : target_idx - int(points_per_freq * 1000)] = 0

new_sig = irfft(yf)

plt.plot(signal, color='#9BD8BA', linewidth=0.3, alpha=0.8)
plt.plot(new_sig, color='#e8736b', linewidth=0.3, alpha=0.8)
plt.show()

