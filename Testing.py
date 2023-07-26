import os
import McsPy.McsData
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
import numpy as np


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
    _ = plt.plot(time_in_sec, signal_in_uV, linewidth=0.2)
    _ = plt.xlabel('Time (%s)' % ureg.s)
    _ = plt.ylabel('Voltage (%s)' % ureg.uV)
    _ = plt.title('Channel %s' % channel_info.info['Label'])
    if show:
        plt.show()


def detect_threshold_falling_crossings(signal, fs, threshold, dead_time):
    """
    Detect threshold crossings in a signal with dead time and return them as an array

    The signal transitions from a sample above the threshold to a sample below the threshold for a detection and
    the last detection has to be more than dead_time apart from the current one.

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold: The threshold for the signal
    :param dead_time: The dead time in seconds.
    """
    dead_time_idx = dead_time * fs
    threshold_crossings = np.diff((signal <= threshold).astype(int) > 0).nonzero()[0]
    distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    while not np.all(distance_sufficient):
        # repeatedly remove all threshold crossings that violate the dead_time
        threshold_crossings = threshold_crossings[distance_sufficient]
        distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    return threshold_crossings


def detect_threshold_rising_crossings(signal, fs, threshold, dead_time):
    """
    Detect threshold crossings in a signal with dead time and return them as an array

    The signal transitions from a sample above the threshold to a sample below the threshold for a detection and
    the last detection has to be more than dead_time apart from the current one.

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold: The threshold for the signal
    :param dead_time: The dead time in seconds.
    """
    dead_time_idx = dead_time * fs
    threshold_crossings = np.diff((signal >= threshold).astype(int) > 0).nonzero()[0]
    distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    while not np.all(distance_sufficient):
        # repeatedly remove all threshold crossings that violate the dead_time
        threshold_crossings = threshold_crossings[distance_sufficient]
        distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    return threshold_crossings


def detect_distance_minmax(spikes, fs, dead_time):
    """
    Work in progress
    Combine and detect sufficient distance between min and max spikes not to recount the same spikes

    :param spikes: spike timing array
    :param fs: The sampling frequency in Hz
    :param dead_time: The dead time in seconds.
    """
    dead_time_idx = dead_time * fs
    unique_crossing = np.sort(spikes)
    distance_sufficient = np.insert(np.diff(unique_crossing) >= dead_time_idx, 0, True)
    while not np.all(distance_sufficient):
        # repeatedly remove all threshold crossings that violate the dead_time
        unique_crossing = unique_crossing[distance_sufficient]
        distance_sufficient = np.insert(np.diff(unique_crossing) >= dead_time_idx, 0, True)

    return unique_crossing


def get_next_minimum(signal, index, max_samples_to_search):
    """
    Returns the index of the next minimum in the signal after an index

    :param signal: The signal as a 1-dimensional numpy array
    :param index: The scalar index
    :param max_samples_to_search: The number of samples to search for a minimum after the index
    """
    search_end_idx = min(index + max_samples_to_search, len(signal))
    min_idx = np.argmin(signal[index:search_end_idx])
    return index + min_idx


def get_next_maximum(signal, index, max_samples_to_search):
    """
    Returns the index of the next maximum in the signal after an index

    :param signal: The signal as a 1-dimensional numpy array
    :param index: The scalar index
    :param max_samples_to_search: The number of samples to search for a minimum after the index
    """
    search_end_idx = min(index + max_samples_to_search, len(signal))
    max_idx = np.argmax(signal[index:search_end_idx])
    return index + max_idx


def align_to_minimum(signal, fs, threshold_crossings, search_range):
    """
    Returns the index of the next negative spike peak for all threshold crossings

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold_crossings: The array of indices where the signal crossed the detection threshold
    :param search_range: The maximum duration in seconds to search for the minimum after each crossing
    """
    search_end = int(search_range * fs)
    aligned_spikes = [get_next_minimum(signal, t, search_end) for t in threshold_crossings]
    return np.array(aligned_spikes)


def align_to_maximum(signal, fs, threshold_crossings, search_range):
    """
    Returns the index of the next negative spike peak for all threshold crossings

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold_crossings: The array of indices where the signal crossed the detection threshold
    :param search_range: The maximum duration in seconds to search for the minimum after each crossing
    """
    search_end = int(search_range * fs)
    aligned_spikes = [get_next_maximum(signal, t, search_end) for t in threshold_crossings]
    return np.array(aligned_spikes)


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

"""
Start of loop/function
"""

noise_mad = np.median(np.absolute(data_array[3])) / 0.6745

falling_threshold = -7 * noise_mad
rising_threshold = 7 * noise_mad

print('Threshold : {0:g} V'.format(rising_threshold))
print('Threshold : {0:g} V'.format(falling_threshold), "\n")

fs = int(stream.channel_infos[ids[3]].sampling_frequency.magnitude)
falling_crossings = detect_threshold_falling_crossings(data_array[3], fs, falling_threshold, 0.003)
rising_crossings = detect_threshold_rising_crossings(data_array[3], fs, rising_threshold, 0.003)
spks_fall = align_to_minimum(data_array[3], fs, falling_crossings, 0.002)
# spks_rise = align_to_maximum(data_array[3], fs, rising_crossings, 0.002)

# print("Rising edge spike count : %s" % len(spks_rise))
print("Falling edge spike count : %s" % len(spks_fall))

# spks = np.sort(np.append(spks_fall, spks_rise))
spks = detect_distance_minmax(spks_fall, fs, 0.0005)

range_in_s = (timeStart, timeStop)

ts_spks = spks / fs
spks_in_range = ts_spks[(ts_spks >= range_in_s[0]) & (ts_spks <= range_in_s[1])]

print("-----------------------------------------------------------")
# frequency
fq = len(spks) / (timeStop - timeStart)
print("Frequency : %s" % round(fq, 2))

print(spks)
spk_int = np.diff(spks)
print(spk_int)

_ = np.std(spk_int) / np.mean(spk_int)
print("Coefficient of variation : %s" % round(_, 2))

array = []
burst_list = []
bursts = []
n = 0

# 50 ms does not return the same bursts as MCS ?
for i in range(len(spk_int)):
    if spk_int[i] < 150:
        j = i
        while spk_int[j] < 150:
            array.append(spks[j])
            j += 1
            if j >= len(spk_int):
                break
        if len(array) >= 4:
            bursts.append(array)
            burst_list += array
            n += 1
        array = []

ts_total = np.asarray(burst_list) / fs
bursts_in_range = ts_total[(ts_total >= range_in_s[0]) & (ts_total <= range_in_s[1])]

plot_analog_stream_channel(stream, 3, from_in_s=timeStart, to_in_s=timeStop, show=False)
_ = plt.plot(spks_in_range, [falling_threshold * 1e6] * spks_in_range.shape[0], 'bo', ms=0.7, alpha=0.5)
# _ = plt.plot([min(bursts[0]) / fs, max(bursts[0]) / fs], [0, 0], 'r-', alpha=0.9, linewidth='5')
# _ = plt.hlines(falling_threshold * 1e6, min(bursts[0]) / fs, max(bursts[0]) / fs, colors="r")

# Range does not work yet
for i in range(len(bursts)):
    # Drawing a line does not work, because the length is too short ;(
    burst_dot = np.arange(min(bursts[i]) / fs, max(bursts[i]) / fs, (max(bursts[i]) / fs - min(bursts[i]) / fs) / 100)
    _ = plt.plot(burst_dot, [2] * len(burst_dot), 'rs', alpha=0.9, ms=0.8)
    _ = plt.plot(min(bursts[i]) / fs, 0, 'ro', ms=0.7, alpha=0.5)

plt.show()
