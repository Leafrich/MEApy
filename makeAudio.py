import os
import numpy as np
import matplotlib.pyplot as plt

import McsPy.McsData

from scipy.io.wavfile import write

# sampling rate
rate = 10000
ts = 1.0 / rate
t = np.arange(0, 300, ts)

electrodeID = 63

# get the signal
rootDir = os.path.dirname(os.path.abspath(__file__))
D5data = f'{rootDir}\\Cx_DIV24.h5'
file = McsPy.McsData.RawData(D5data)
electrode_stream = file.recordings[0].analog_streams[0]
ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
signal = electrode_stream.get_channel_in_range(ids[electrodeID], 0, len(t) - 1)

signalArray = np.array(signal[0])

signalIntArray = (np.ceil(signalArray)).astype(int)
write('cxAudio.mp3', rate, signalArray * 10000)

plt.figure(figsize=(16, 5))
plt.plot(t, signalArray, 'g', linewidth=0.05)
plt.ylabel('Amplitude')

plt.xlabel('Time (s)')
plt.show()
