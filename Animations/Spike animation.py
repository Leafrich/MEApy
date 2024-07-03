import os
import McsPy.McsData
from Functions import *

electrode_id = 5
timeStart = 0
timeStop = 100

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
# spks_rise = align_to_maximum(signal, fs, rising_crossings, 0.002)

# print("Rising edge spike count : %s" % len(spks_rise))
print("Falling edge spike count : %s" % len(spks_fall))

# spks = np.sort(np.append(spks_fall, spks_rise))
spks = detect_distance_minmax(spks_fall, fs, 0.003)

range_in_s = (timeStart, timeStop)
ts_spks = spks / fs
spks_in_range = ts_spks[(ts_spks >= range_in_s[0]) & (ts_spks <= range_in_s[1])]

pre = 0.002
post = 0.003
cutouts = extract_waveforms(signal, fs, spks, pre, post)
print("Spike count : " + str(len(cutouts)))

print("-----------------------------------------------------------")

import matplotlib.animation as animation

fig, ax = plt.subplots()
time_in_us = np.arange(-pre * 1000, post * 1000, 1e3 / fs)
avspike = np.mean(cutouts, axis=0)

count = len(cutouts)
d = {}
for i in range(count):
    d["line{0}".format(i)] = ax.plot(time_in_us[0], cutouts[i][0], color='#7abcd6', linewidth=0.5, alpha=0.8)[0]

avline = ax.plot(time_in_us[0], avspike[0], color='#fc9d03')[0]

ax.set(xlim=[(-pre + (-pre * .1)) * 1e3, (post + (post * .1)) * 1e3],
       ylim=[np.min(cutouts) + (np.min(cutouts) * .1), np.max(cutouts) + (np.max(cutouts) * .1)],
       xlabel='Time [ms]', ylabel='µV')

def update(frame):
    # for each frame, update the data stored on each artist.
    # update the line plot:
    for i in range(count):
        d["line{0}".format(i)].set_xdata(time_in_us[:frame])
        d["line{0}".format(i)].set_ydata(cutouts[i][:frame])
    avline.set_xdata(time_in_us[:frame])
    avline.set_ydata(avspike[:frame])
    return avline, ["line{0}".format(i) for i in range(count)]

ani = animation.FuncAnimation(fig=fig, func=update, frames=int(len(cutouts[0])), interval=10)
ani.save("Spike animation.mp4")
plt.show()

# Y axis is incorrect