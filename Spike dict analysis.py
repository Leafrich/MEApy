import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)

timeStart = 0
timeStop = 300
range_in_s = (timeStart, timeStop)

rootDir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f'{rootDir}\\Hippocampus spk analysis.csv')
df = df.iloc[:, [1, 3, 6, 9]]

uniNames = pd.unique(df['Experiment'])
totalDIVs = ['DIV4', 'DIV6', 'DIV8', 'DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24', 'DIV26', 'DIV30']
DIVarr = ['DIV4', 'DIV6', 'DIV8', 'DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24']  # in future experiments this could become uni val "experiment"
Dict = {}

# df['ID'] = df['Well Label'].astype(str) + df['Channel Label'].astype(str)  # not needed, remove in time when replaced

for _ in range(len(uniNames)):
    Dict[DIVarr[_]] = df.loc[df['Experiment'] == uniNames[_], :]

# # Spk frequency overtime
# spk_rate_array = []
# temp_array = []
# temp_err_array = []
# yerr = []
# for i in DIVarr:  # DIV seperating
#     burst_array = []
#     day = Dict[i]
#     for w in pd.unique(Dict[str(i)]["Well Label"]):  # Well seperation
#         well = day.loc[day['Well Label'] == w, :]
#         for elec in pd.unique(well["Channel Label"]):  # change to right object
#             elecdata = well.loc[well['Channel Label'] == elec, :]
#             spks_in_range = elecdata.loc[elecdata['Timestamp [µs]'] > timeStart * 1e6, :]  # needs to be done in one line
#             spks_in_range = spks_in_range.loc[spks_in_range['Timestamp [µs]'] < timeStop * 1e6, :]
#
#             temp_array.append(len(spks_in_range)/300)
#             if len(temp_array) > 1:
#                 temp_err_array = np.std(temp_array)
#     yerr.append(temp_err_array)
#     spk_rate_array.append(np.mean(temp_array))
#     temp_array = []
#     temp_err_array = []
#
# plt.title("Spike frequency")  # average burst rate of all wells/day
# plt.ylabel("Frequency (Hz)")
# plt.bar(DIVarr, spk_rate_array, color='#99e084')
# plt.errorbar(DIVarr, spk_rate_array, yerr, color='black', lw=1, alpha=0.4, capsize=3, capthick=0.7, fmt='none')
# plt.ylim(bottom=0)
# plt.show()
#
# # active channels
# act_channels = []
# div_act = []
# for i in DIVarr:  # DIV seperating
#     burst_array = []
#     day = Dict[i]
#     for w in pd.unique(Dict[str(i)]["Well Label"]):  # Well seperation
#         well = day.loc[day['Well Label'] == w, :]
#         act_channels.append(len(pd.unique(well["Channel Label"])))
#     div_act.append(np.mean(act_channels))
#     act_channels = []
#
# # barplot of active channels of all wells per day
# plt.title("Active channel of all wells/day")
# plt.ylabel("Active channels/well")
#
# plt.bar(DIVarr, div_act, color='#FFAF4A')
# plt.show()

# CV of spks
temp = []
CVs = []
for i in DIVarr:  # DIV seperating
    burst_array = []
    day = Dict[i]
    for w in pd.unique(Dict[str(i)]["Well Label"]):  # Well seperation
        well = day.loc[day['Well Label'] == w, :]
        for elec in pd.unique(well["Channel Label"]):  # change to right object
            elecdata = well.loc[well['Channel Label'] == elec, :]
            spks_in_range = elecdata.loc[elecdata['Timestamp [µs]'] > timeStart * 1e6, :]  # needs to be done in one line
            spks_in_range = spks_in_range.loc[spks_in_range['Timestamp [µs]'] < timeStop * 1e6, :]

            spk_int = np.diff(spks_in_range['Timestamp [µs]'])
            if np.std(spk_int) != float or np.mean(spk_int) != float:  # this doesnt work, nan screw with the std
                print(i, w, elec)
                continue
            temp.append(np.std(spk_int) / np.mean(spk_int))

    CVs.append(np.mean(temp))
    temp = []

# barplot of active channels of all wells per day
plt.title("Active channel of all wells/day")
plt.ylabel("Active channels/well")

plt.bar(DIVarr, CVs, color='#FFAF4A')
plt.show()

