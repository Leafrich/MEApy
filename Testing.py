import os
import McsPy.McsData

rootDir = os.path.dirname(os.path.abspath(__file__))
D5data = f'{rootDir}\\20230530_Cortex_pMEA.h5'
data = McsPy.McsData.RawData(D5data)
stream = data.recordings[0].analog_streams[0]
ids = [c.channel_id for c in stream.channel_infos.values()]  # needed to find proper channel_data rows for .get_channel

# Methods extraction, mild succes .channel_infos not found?
# _ = [method for method in dir(stream) if callable(getattr(stream, method))]
# for i in range(len(_)):
#     print(_[i])

# Finding info columns works well
# well_ids = [c.group_id for c in stream.channel_infos.values()]
# print(well_ids)

electrode_channel_data = stream.get_channel(ids[0])
print(electrode_channel_data)
