import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

import McsPy.McsData
from McsPy import ureg, Q_

# sampling rate
rate = 10000

ts = 1.0 / rate
t = np.arange(0, 10, ts)

# get the signal
rootDir = os.path.dirname(os.path.abspath(__file__))
D5data = f'{rootDir}\\wD6_D18_hc.h5'
file = McsPy.McsData.RawData(D5data)
electrode_stream = file.recordings[0].analog_streams[0]
ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
signal = electrode_stream.get_channel_in_range(ids[7], 0, len(t)-1)
signalArray = np.array(signal[0])

y = list()
freq = 0

for i in range(0, len(t)):
    c = np.sin(freq * 2 * np.pi)
    y.append(c)

    freq = freq + 2 * np.pi * signalArray[i] * ts * 10000000

sinSignal = np.array(y)
print(sinSignal[12])
write('test.wav', rate, sinSignal)

plt.figure(figsize = (8, 8))
plt.subplot(211)
plt.plot(t, sinSignal, 'b')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.plot(t, signalArray, 'b')
plt.ylabel('Amplitude')

plt.xlabel('Time (s)')
plt.show()
