import os
import McsPy.McsData
from Functions import *

electrode_id = 4
timeStart = 0
timeStop = 300

rootDir = os.path.dirname(os.path.abspath(os.path.join(__file__ ,"..")))
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

plot_analog_stream_channel(electrode_stream, electrode_id, from_in_s=timeStart, to_in_s=timeStop, show=False)
_ = plt.plot(falling_in_range, [falling_threshold * 1e6] * falling_in_range.shape[0], 'bo', ms=0.4)
_ = plt.plot(rising_in_range, [rising_threshold * 1e6] * rising_in_range.shape[0], 'ro', ms=0.4)
plt.show()

pre = 0.002
post = 0.002
cutouts = extract_waveforms(signal, fs, spks, pre, post)
print("Spike count : " + str(len(cutouts)))
print(np.shape(cutouts))

print("-----------------------------------------------------------")

plot_waveforms(cutouts, fs, pre, post, n=500)
plt.show()

