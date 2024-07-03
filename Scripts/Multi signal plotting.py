from Functions import plot_well
import os
import McsPy.McsData

# MCS24Dict = {
#     "0": 0, "1": z, "2": z, "3": 0,
#     "4": z, "5": z, "6": z, "7": z,
#     "8": z, "9": z, "10": z, "11": z,
#     "12": 0, "13": z, "14": z, "15": 0,
# }

well_id = 1
electrode_id = 6
timeStart = 0
timeStop = 30

rootDir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
D5data = f'{rootDir}\\Data\\Cx_DIV24.h5'
file = McsPy.McsData.RawData(D5data)
electrode_stream = file.recordings[0].analog_streams[0]
ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
channel_id = ids[electrode_id]

info = electrode_stream.channel_infos[channel_id].info

print("-----------------------------------------------------------")
print("Sampling frequency : %s Hz" % int(electrode_stream.channel_infos[channel_id].sampling_frequency.magnitude))
print("Bandwidth : %s - %s Hz" % (info['HighPassFilterCutOffFrequency'], info['LowPassFilterCutOffFrequency']))
print("-----------------------------------------------------------")

plot_well(electrode_stream, from_in_s=timeStart, to_in_s=timeStop, show=True)


