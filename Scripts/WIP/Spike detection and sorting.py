import os
import McsPy.McsData
from McsPy import ureg, Q_
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection


def plot_analog_stream_channel(analog_stream, channel_idx, from_in_s=0, to_in_s=None, show=False):
    """
    Plots data from a single AnalogStream channel

    :param analog_stream: A AnalogStream object
    :param channel_idx: A scalar channel index (0 <= channel_idx < # channels in the AnalogStream)
    :param from_in_s: The start timestamp of the plot (0 <= from_in_s < to_in_s). Default: 0
    :param to_in_s: The end timestamp of the plot (from_in_s < to_in_s <= duration). Default: None (= recording duration)
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
    _ = plt.plot(time_in_sec, signal_in_uV, linewidth=0.1)
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
    search_end_idx = min(index + max_samples_to_search, signal.shape[0])
    min_idx = np.argmin(signal[index:search_end_idx])
    return index + min_idx


def get_next_maximum(signal, index, max_samples_to_search):
    """
    Returns the index of the next maximum in the signal after an index

    :param signal: The signal as a 1-dimensional numpy array
    :param index: The scalar index
    :param max_samples_to_search: The number of samples to search for a minimum after the index
    """
    search_end_idx = min(index + max_samples_to_search, signal.shape[0])
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


def extract_waveforms(signal, fs, spikes_idx, pre, post):
    """
    Extract spike waveforms as signal cutouts around each spike index as a spikes x samples numpy array

    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param spikes_idx: The sample index of all spikes as a 1-dim numpy array
    :param pre: The duration of the cutout before the spike in seconds
    :param post: The duration of the cutout after the spike in seconds
    """
    cutouts = []
    pre_idx = int(pre * fs)
    post_idx = int(post * fs)
    for index in spikes_idx:
        if index - pre_idx >= 0 and index + post_idx <= signal.shape[0]:
            cutout = signal[(index - pre_idx):(index + post_idx)]
            cutouts.append(cutout)
    return np.stack(cutouts)


def smooth(y, str):
    """
    Plot an overlay of spike cutouts

    :param y: array of signal
    :param str: between range of 1-8
    """

    box = np.ones(str) / str
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_waveforms(cutouts, fs, pre, post, n=100, color='k', show=True):
    """
    Plot an overlay of spike cutouts

    :param cutouts: A spikes x samples array of cutouts
    :param fs: The sampling frequency in Hz
    :param pre: The duration of the cutout before the spike in seconds
    :param post: The duration of the cutout after the spike in seconds
    :param n: The number of cutouts to plot, or None to plot all. Default: 100
    :param color: The line color as a pyplot line/marker style. Default: 'k'=black
    :param show: Set this to: False to disable showing the plot. Default: True
    """
    if n is None:
        n = cutouts.shape[0]
    n = min(n, cutouts.shape[0])
    time_in_us = np.arange(-pre * 1000, post * 1000, 1e3 / fs)
    if show:
        _ = plt.figure(figsize=(12, 6))

    for i in range(n):
        _ = plt.plot(time_in_us, smooth(cutouts[i,], 2) * 1e6, color, linewidth=0.9, alpha=0.5)
        _ = plt.xlabel('Time (%s)' % ureg.ms)
        _ = plt.ylabel('Voltage (%s)' % ureg.uV)
        _ = plt.title('Cutouts')

    if show:
        plt.show()


electrode_id = 4
timeStart = 0
timeStop = 300

rootDir = os.path.dirname(os.path.abspath(os.path.join(__file__, "../..")))
D5data = f'{rootDir}\\Data\\Cx_50kHz.h5'
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

# for i in range(len(ids)):
#     plot_analog_stream_channel(electrode_stream, i, from_in_s=timeStart, to_in_s=timeStop, show=False)
#     plt.show()

noise_mad = np.median(np.absolute(signal)) / 0.6745

falling_threshold = -5 * noise_mad
rising_threshold = 5 * noise_mad

print('Threshold : {0:g} V'.format(rising_threshold))
print('Threshold : {0:g} V'.format(falling_threshold), "\n")

fs = int(electrode_stream.channel_infos[channel_id].sampling_frequency.magnitude)
falling_crossings = detect_threshold_falling_crossings(signal, fs, falling_threshold, 0.003)
rising_crossings = detect_threshold_rising_crossings(signal, fs, rising_threshold, 0.003)
spks_fall = align_to_minimum(signal, fs, falling_crossings, 0.002)
spks_rise = align_to_maximum(signal, fs, rising_crossings, 0.002)

print("Rising edge spike count : %s" % len(spks_rise))
print("Falling edge spike count : %s" % len(spks_fall))

spks = np.sort(np.append(spks_fall, spks_rise))
spks = detect_distance_minmax(spks, fs, 0.003)

ts_rise = spks_rise / fs
ts_fall = spks_fall / fs
range_in_s = (timeStart, timeStop)
falling_in_range = ts_fall[(ts_fall >= range_in_s[0]) & (ts_fall <= range_in_s[1])]
rising_in_range = ts_rise[(ts_rise >= range_in_s[0]) & (ts_rise <= range_in_s[1])]

ts_spks = spks / fs
spks_in_range = ts_spks[(ts_spks >= range_in_s[0]) & (ts_spks <= range_in_s[1])]

plot_analog_stream_channel(electrode_stream, electrode_id, from_in_s=timeStart, to_in_s=timeStop, show=False)
_ = plt.plot(spks_in_range, [falling_threshold * 1e6] * spks_in_range.shape[0], 'bo', ms=0.4)
_ = plt.plot(rising_in_range, [rising_threshold * 1e6] * rising_in_range.shape[0], 'ro', ms=0.4)
plt.show()

pre = 0.002
post = 0.002
cutouts = extract_waveforms(signal, fs, spks, pre, post)
print("Spike count : " + str(len(cutouts)))
print(np.shape(cutouts))

print("-----------------------------------------------------------")

plot_waveforms(cutouts, fs, pre, post, n=500)

# min_amplitude = np.amin(cutouts, axis=1)
# max_amplitude = np.amax(cutouts, axis=1)
#
# _ = plt.figure(figsize=(8, 8))
# _ = plt.plot(min_amplitude * 1e6, max_amplitude * 1e6, '.')
# _ = plt.xlabel('Min. Amplitude (%s)' % ureg.uV)
# _ = plt.ylabel('Max. Amplitude (%s)' % ureg.uV)
# _ = plt.title('Min/Max Spike Amplitudes')
#
# plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# PCA analysis
#----------------------------------------------------------------------------------------------------------------------#

scaler = StandardScaler()
scaled_cutouts = scaler.fit_transform(cutouts)

pca = PCA()
pca.fit(scaled_cutouts)

# j = 0
# n = 0
# for i in range(len(pca.explained_variance_ratio_)):
#     j += pca.explained_variance_ratio_[i]
#     n += 1
#     if j >= .2:
#         break

transformed_3d = pca.fit_transform(scaled_cutouts)
print("Dim transformed %s" % str(np.shape(transformed_3d)))

# _ = plt.figure(figsize=(15, 5))
# _ = plt.subplot(1, 3, 1)
# _ = plt.plot(transformed_3d[:, 0], transformed_3d[:, 1], '.')
# _ = plt.xlabel('Principal Component 1')
# _ = plt.ylabel('Principal Component 2')
# _ = plt.title('PC1 vs PC2')
# _ = plt.subplot(1, 3, 2)
# _ = plt.plot(transformed_3d[:, 0], transformed_3d[:, 2], '.')
# _ = plt.xlabel('Principal Component 1')
# _ = plt.ylabel('Principal Component 3')
# _ = plt.title('PC1 vs PC3')
# _ = plt.subplot(1, 3, 3)
# _ = plt.plot(transformed_3d[:, 1], transformed_3d[:, 2], '.')
# _ = plt.xlabel('Principal Component 2')
# _ = plt.ylabel('Principal Component 3')
# _ = plt.title('PC2 vs PC3')
# plt.show()

# print("%s Compenents 0.6 variabilty explained" % n)
n_components = 3

gmm = GaussianMixture(n_components=n_components, n_init=10)
labels = gmm.fit_predict(transformed_3d)

_ = plt.figure(figsize=(8, 8))
for i in range(n_components):
    idx = labels == i
    _ = plt.plot(transformed_3d[idx, 0], transformed_3d[idx, 1], '.')
    _ = plt.title('Cluster assignments by a GMM')
    _ = plt.xlabel('Principal Component 1')
    _ = plt.ylabel('Principal Component 2')
    _ = plt.axis('tight')
plt.show()

_ = plt.figure(figsize=(12, 6))
for i in range(n_components):
    idx = labels == i
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
    plot_waveforms(cutouts[idx, :], fs, pre, post, n=100, color=color, show=False)
plt.show()
