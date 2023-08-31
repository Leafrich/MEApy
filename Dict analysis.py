import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def in_range(pd_data):
    burst_in_range = []
    for k in range(len(pd_data['Channel Label'])):
        if range_in_s[1] >= pd_data['Start timestamp [µs]'].iloc[k] / 1e6 >= range_in_s[0]:  # start within range
            if pd_data['End timestamp [µs]'].iloc[k] / 1e6 < range_in_s[1]:  # end within range
                burst_in_range.append([pd_data['Start timestamp [µs]'].iloc[k],
                                       pd_data['End timestamp [µs]'].iloc[k]])
            else:  # start in range/ end not in range
                burst_in_range.append([pd_data['Start timestamp [µs]'].iloc[k],
                                       timeStop * 1e6])
                break
        else:  # start not in range
            if range_in_s[1] >= pd_data['End timestamp [µs]'].iloc[k] / 1e6 >= range_in_s[0]:  # end in range
                burst_in_range.append([timeStart * 1e6, pd_data['End timestamp [µs]'].iloc[k]])
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


pd.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)

timeStart = 0
timeStop = 300
range_in_s = (timeStart, timeStop)

rootDir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f'{rootDir}\\Hippocampus bursts analysis.csv')
df = df.iloc[:, [1, 3, 6, 9, 10, 11, 12]]
df['End timestamp [µs]'] = df['Start timestamp [µs]'] + df['Duration [µs]']

uniNames = pd.unique(df['Experiment'])
totalDIVs = ['DIV4', 'DIV6', 'DIV8', 'DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24', 'DIV26', 'DIV30']
DIVarr = ['DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24']  # in future experiments this could become uni val "experiment"
Dict = {}

# df['ID'] = df['Well Label'].astype(str) + df['Channel Label'].astype(str)  # not needed, remove in time when replaced

for _ in range(len(uniNames)):
    Dict[DIVarr[_]] = df.loc[df['Experiment'] == uniNames[_], :]

# # Burst Hz
# burst_rate_array = [0, 0, 0]  # lazy start for burst free days
# temp_array = []
# temp_err_array = []  # lazy start for burst free days
# yerr = [0, 0, 0]
# for i in DIVarr:  # DIV seperating
#     burst_array = []
#     day = Dict[i]
#     for w in pd.unique(Dict[str(i)]["Well Label"]):  # Well seperation
#         well = day.loc[day['Well Label'] == w, :]
#         for elec in pd.unique(well["Channel Label"]):  # change to right object
#             elecdata = well.loc[well['Channel Label'] == elec, :]
#             burst_in_range = in_range(elecdata)
#             # x = np.asarray(list(chain.from_iterable(burst_in_range)))  # flattening could also be done differently
#             # plt.scatter(x / 1e6, [str(elec)] * len(x), color='#FFAF4A', marker='s', zorder=2, s=2)
#
#             y = sep_bursts(burst_in_range)
#             temp_array.append(len(y)/300)
#             if len(temp_array) > 1:
#                 temp_err_array = np.std(temp_array)
#
#     yerr.append(temp_err_array)
#     burst_rate_array.append(np.mean(temp_array))
#     temp_array = []
#     temp_err_array = []
#
# plt.title("Average burst rate")  # average burst rate of all wells/day
# plt.ylabel("Burst rate (Hz)")
# burst_rate_array.append(0)  # lazy append for burst free days
# burst_rate_array.append(0)
# yerr.append(0)  # lazy append for burst free days
# yerr.append(0)
# plt.bar(totalDIVs, burst_rate_array, color='#99e084')
# plt.errorbar(totalDIVs, burst_rate_array, yerr, color='black', lw=1, alpha=0.4, capsize=3, capthick=0.7, fmt='none')
# plt.ylim(bottom=0)
# plt.show()

# # Burst CV calculation
# burst_cv_array = [0, 0, 0]  # lazy start for burst free days
# temp = []
# temp_array = []
# for i in DIVarr:  # DIV seperating
#     burst_array = []
#     day = Dict[i]
#     for w in pd.unique(Dict[str(i)]["Well Label"]):  # Well seperation
#         well = day.loc[day['Well Label'] == w, :]
#         for elec in pd.unique(well["Channel Label"]):  # change to right object
#             elecdata = well.loc[well['Channel Label'] == elec, :]
#             burst_in_range = in_range(elecdata)
#             y = sep_bursts(burst_in_range)
#             for bur in range(len(y) - 1):
#                 temp_array.append(y[bur + 1][0][0] - y[bur][-1][1])
#             if len(temp_array) > 1:
#                 cv = np.std(temp_array)/np.mean(temp_array)
#                 temp.append(cv)
#     burst_cv_array.append(np.mean(temp))
#     temp_array = []
#
# plt.title("Average burst CV")  # average burst rate of all wells/day
# plt.ylabel("CV (std of interval/mean of interval)")
# burst_cv_array.append(0)  # lazy append for burst free days
# burst_cv_array.append(0)
# plt.bar(totalDIVs, burst_cv_array, color='#99e084')
# plt.ylim(bottom=0)
# plt.show()


# # Active bursting channels
# act_channels = []
# div_act = [0, 0, 0]  # lazy start for burst free days
# for i in DIVarr:  # DIV seperating
#     burst_array = []
#     day = Dict[i]
#     for w in pd.unique(Dict[str(i)]["Well Label"]):  # Well seperation
#         # plt.title(str(i) + ' ' + str(w))
#         well = day.loc[day['Well Label'] == w, :]
#         act_channels.append(len(pd.unique(well["Channel Label"])))
#     div_act.append(np.mean(act_channels))
#     act_channels = []
#
# # barplot of active channels of all wells per day
# plt.title("Active channel of all wells/day")
# plt.ylabel("Active channels/well")
# div_act.append(0)  # lazy append for burst free days
# div_act.append(0)
# plt.bar(totalDIVs, div_act, color='#FFAF4A')
# plt.show()
