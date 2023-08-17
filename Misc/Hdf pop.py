import os
import McsPy.McsData
import matplotlib.pyplot as plt
import numpy as np

timeStart = 0
timeStop = 120

rootDir = os.path.dirname(os.path.abspath(__file__))
D5data = f'{rootDir}\\wD6_D18_hc.h5'
file = McsPy.McsData.RawData(D5data)
electrode_stream = file.recordings[0].analog_streams[0]  # All data streams (incl wells and electrodes)

print(electrode_stream)

# is this needed, considering I'm using all streams
ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
channel_id = ids[0]

channel_info = electrode_stream.channel_infos[channel_id]
sampling_frequency = channel_info.sampling_frequency.magnitude

info = electrode_stream.channel_infos[channel_id].info

print("-----------------------------------------------------------")
print("Sampling frequency : %s Hz" % int(sampling_frequency))
print("Bandwidth : %s - %s Hz" % (info['HighPassFilterCutOffFrequency'], info['LowPassFilterCutOffFrequency']))
print("-----------------------------------------------------------")

from_idx = max(0, int(timeStart * sampling_frequency))
if timeStop is None:
    to_idx = electrode_stream.channel_data.shape[1]
else:
    to_idx = min(electrode_stream.channel_data.shape[1], int(timeStop * sampling_frequency))

lenSig = electrode_stream.get_channel_in_range(ids[0], from_idx, to_idx)



# adapt size of array to maximum streams available
sigArray = np.empty(shape=(12, len(lenSig[0])))

for j in range(12):
    signal = electrode_stream.get_channel_in_range(ids[j], from_idx, to_idx)
    sigArray[j] = signal[0]
