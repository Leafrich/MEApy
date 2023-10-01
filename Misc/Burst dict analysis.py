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
totalDIVs = ['DIV4', 'DIV6', 'DIV8', 'DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24']
DIVarr = ['DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24']  # in future experiments this could become uni val "experiment"
Dict = {}

# df['ID'] = df['Well Label'].astype(str) + df['Channel Label'].astype(str)  # not needed, remove in time when replaced

for _ in range(len(uniNames)):
    Dict[DIVarr[_]] = df.loc[df['Experiment'] == uniNames[_], :]

# Burst Hz
burst_rate_array = [0, 0, 0]  # lazy start for burst free days
temp_array = []
temp_err_array = []  # lazy start for burst free days
yerr = [0, 0, 0]
for i in DIVarr:  # DIV seperating
    burst_array = []
    day = Dict[i]
    for w in pd.unique(Dict[str(i)]["Well Label"]):  # Well seperation
        well = day.loc[day['Well Label'] == w, :]
        for elec in pd.unique(well["Channel Label"]):  # change to right object
            elecdata = well.loc[well['Channel Label'] == elec, :]
            burst_in_range = in_range(elecdata)
            # x = np.asarray(list(chain.from_iterable(burst_in_range)))  # flattening could also be done differently
            # plt.scatter(x / 1e6, [str(elec)] * len(x), color='#FFAF4A', marker='s', zorder=2, s=2)

            y = sep_bursts(burst_in_range)
            temp_array.append(len(y)/300)
            if len(temp_array) > 1:
                temp_err_array = np.std(temp_array)

    yerr.append(temp_err_array)
    burst_rate_array.append(np.mean(temp_array))
    temp_array = []
    temp_err_array = []

plt.title("Average burst rate")  # average burst rate of all wells/day
plt.ylabel("Burst rate (Hz)")
plt.bar(totalDIVs, burst_rate_array, color='#99e084', edgecolor='#444444')
plt.errorbar(totalDIVs, burst_rate_array, yerr, color='black', lw=1, alpha=0.4, capsize=3, capthick=0.7, fmt='none')
plt.ylim(bottom=0)
plt.show()

# # Burst CV bargraph
# burst_cv_array = [0, 0, 0]  # lazy start for burst free days
# temp = []
# temp_array = []
# yerr = [0, 0, 0]
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
#     yerr.append(np.std(temp))
#     temp_array = []
#     temp = []
#
# plt.title("Average burst CV")  # average burst rate of all wells/day
# plt.ylabel("CV (std of interval/mean of interval)")
# plt.bar(totalDIVs, burst_cv_array, color='#99e084')
# plt.errorbar(totalDIVs, burst_cv_array, yerr, color='black', lw=1, alpha=0.4, capsize=3, capthick=0.7, fmt='none')
# plt.ylim(bottom=0)
# plt.show()

# Burst CV linegraph
cvdf = pd.DataFrame(columns=["Electrode", "Well", "DIV", "CV"])
eDay = [0, 0, 0]
burst_cv_array = []  # lazy start for burst free days
temp = []
temp_array = []
yerr = [0, 0, 0]
for i in DIVarr:  # DIV seperating
    burst_array = []
    day = Dict[i]
    for w in pd.unique(Dict[str(i)]["Well Label"]):  # Well seperation
        well = day.loc[day['Well Label'] == w, :]
        for elec in pd.unique(well["Channel Label"]):  # change to right object
            elecdata = well.loc[well['Channel Label'] == elec, :]  # select rows only with right electrode label
            burst_in_range = in_range(elecdata)  # select rows only in given range
            y = sep_bursts(burst_in_range)  # seperate bursting trains events

            for bur in range(len(y) - 1):
                temp_array.append(y[bur + 1][0][0] - y[bur][-1][1])  # calculate interval between bursts

            if len(temp_array) > 1:  # cant calculate a std if there array is 1
                cv = np.std(temp_array)/np.mean(temp_array)
                list_row = [elec, w, i, cv]
                cvdf.loc[len(cvdf)] = list_row
    temp_array = []

# Need to start using pds to make my graphs
plt.title("Average burst CV")  # average burst rate of all wells/day
plt.ylabel("Coefficent of Variance")
for q in pd.unique(cvdf["Well"]):
    x = cvdf.loc[cvdf["Well"] == q]
    for _ in pd.unique(cvdf["Electrode"]):
        y = x.loc[cvdf["Electrode"] == _]
        plt.plot(y["DIV"], y["CV"])

plt.ylim(bottom=0)
plt.show()


# Active bursting channels
act_channels = []
div_act = [0, 0, 0]  # lazy start for burst free days
yerr = [0, 0, 0]
for i in DIVarr:  # DIV seperating
    burst_array = []
    day = Dict[i]
    for w in pd.unique(Dict[str(i)]["Well Label"]):  # Well seperation
        # plt.title(str(i) + ' ' + str(w))
        well = day.loc[day['Well Label'] == w, :]
        act_channels.append(len(pd.unique(well["Channel Label"])))
    div_act.append(np.mean(act_channels))
    yerr.append(np.std(act_channels))
    act_channels = []

# barplot of active channels of all wells per day
plt.title("Active bursting channels")
plt.ylabel("Active channels/well")
plt.bar(totalDIVs, div_act, color='#99e084')
plt.errorbar(totalDIVs, div_act, yerr, color='black', lw=1, alpha=0.4, capsize=3, capthick=0.7, fmt='none')
plt.show()

# ------------------------------------------
# From Main
# ------------------------------------------
# # Burst CV linegraph
# cvdf = pd.DataFrame(columns=["Electrode", "Well", "DIV", "CV"])
# # works without the dict, not sure what is easier though
# for d in pd.unique(bstdf["Experiment"]):  # DIV seperating
#     burst_array = []
#     day = bstdf.loc[bstdf['Experiment'] == d, :]
#     temp = []
#     for w in pd.unique(bstdf["Well Label"]):  # Well seperation
#         well = day.loc[day['Well Label'] == w, :]
#         for e in pd.unique(well["Channel Label"]):  # change to right object
#             elecdata = well.loc[well['Channel Label'] == e, :]  # select rows only with right electrode label
#             burst_in_range = in_range(elecdata)  # select rows only in given range
#             y = sep_bursts(burst_in_range)  # seperate bursting trains events
#
#             # Burst CV Calculations
#             for bur in range(len(y) - 1):
#                 temp.append(y[bur + 1][0][0] - y[bur][-1][1])  # calculate interval between bursts
#             if len(temp) > 1:  # cant calculate a std if the len(array) <= 1
#                 cv = np.std(temp) / np.mean(temp)
#                 list_row = [e, w, d, cv]
#                 cvdf.loc[len(cvdf)] = list_row
#             else:
#                 list_row = [e, w, d, 0]
#                 cvdf.loc[len(cvdf)] = list_row
#
# plt.title("Average burst CV")  # average burst rate of all wells/day
# plt.ylabel("Coefficent of Variance")
# c = 0
# for q in pd.unique(cvdf["Well"]):
#     x = cvdf.loc[cvdf["Well"] == q]
#     for _ in pd.unique(cvdf["Electrode"]):
#         line = x.loc[x["Electrode"] == _]
#         line = line.sort_values(by=["DIV"])  # symptom of adding empty days, otherwise order is messed up
#         plt.plot(line["DIV"], line["CV"], color=wellrainbow[c], alpha=0.3, lw=0.8)
#     c += 1
# plt.ylim(bottom=0)
# plt.tight_layout()
#
# plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# from old main
# Line plot
# plt.title("Burst Frequency")  # average burst rate of all wells/day
# plt.ylabel("Frequency [Hz]")
# c = 0
# for q in pd.unique(dbstdf["Well"]):
#     x = dbstdf.loc[dbstdf["Well"] == q]
#     for _ in pd.unique(dbstdf["Electrode"]):
#         line = x.loc[x["Electrode"] == _]
#         line = line.sort_values(by=["DIV"])  # symptom of adding empty days, otherwise order is messed up
#         plt.plot(line["DIV"], line["Hz"], color=wellrainbow[c], alpha=0.3, lw=0.8)
#     c += 1
# plt.ylim(bottom=0)
# plt.tight_layout()
# plt.show()
#----------------------------------------------------------------------------------------------------------------------#
# # Bar plot
# plt.title("Burst Frequency")  # average burst rate of all wells/day
# plt.ylabel("Frequency [Hz]")
# Hzdf = pd.DataFrame(columns=["DIV", "Mean", "Err"])
# for q in pd.unique(dbstdf["DIV"]):
#     x = dbstdf.loc[dbstdf["DIV"] == q]
#     for _ in pd.unique(dbstdf["Well"]):
#         well = x.loc[x["Well"] == _]
#         list_row = [q, np.mean(well["Hz"]), np.std(well["Hz"])]
#         Hzdf.loc[len(Hzdf)] = list_row
# Hzdf = Hzdf.sort_values(by=["DIV"])  # symptom of adding empty days, otherwise order is messed up
# plt.bar(Hzdf["DIV"], Hzdf["Mean"], color='#FFF8e7', edgecolor='#444444')
# # plt.errorbar(Hzdf["DIV"], Hzdf["Mean"], Hzdf["Err"], color='black', lw=1, alpha=0.4, capsize=3, capthick=0.7, fmt='none')
# plt.ylim(bottom=0)
# plt.tight_layout()
# plt.show()