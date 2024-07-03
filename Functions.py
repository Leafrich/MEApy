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
    _ = plt.plot(time_in_sec, signal_in_uV, color='#94D9A2', linewidth=0.4)
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
        _ = plt.plot(time_in_us, cutouts[i,] * 1e6, color, linewidth=0.9, alpha=0.5)
        _ = plt.xlabel('Time (%s)' % ureg.ms)
        _ = plt.ylabel('Voltage (%s)' % ureg.uV)
        _ = plt.title('Cutouts')

    return # Array of plots


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

        def in_range(pd_data, range_in_s):
            burst_in_range = []
            for k in range(len(pd_data['Channel Label'])):
                if range_in_s[1] >= pd_data['Start timestamp [µs]'].iloc[k] / 1e6 >= range_in_s[
                    0]:  # start within range
                    if pd_data['End timestamp [µs]'].iloc[k] / 1e6 < range_in_s[1]:  # end within range
                        burst_in_range.append([pd_data['Start timestamp [µs]'].iloc[k],
                                               pd_data['End timestamp [µs]'].iloc[k]])
                    else:  # start in range/ end not in range
                        burst_in_range.append([pd_data['Start timestamp [µs]'].iloc[k],
                                               range_in_s[1] * 1e6])
                        break
                else:  # start not in range
                    if range_in_s[1] >= pd_data['End timestamp [µs]'].iloc[k] / 1e6 >= range_in_s[0]:  # end in range
                        burst_in_range.append([range_in_s[0] * 1e6, pd_data['End timestamp [µs]'].iloc[k]])
            return burst_in_range

        def sep_bursts(data):

            bur_int = [j for sub in data for j in sub]  # flatten data into 1D
            bur_int = np.diff(bur_int)  # calculate diff between bursts
            slBurst = []  # new burst interval array
            for n in range(len(bur_int)):  # seperate burst- and interburst intervals
                if n % 2 != 0:
                    slBurst.append(bur_int[n])

            array = []  # temp storage array
            sep_bur = []
            for m in range(int(len(slBurst))):
                array.append(data[m])
                if slBurst[m] > 500000:
                    sep_bur.append(array)
                    array = []
            return sep_bur
            # burst_array.append(sep_bur)