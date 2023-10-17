import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)


#----------------------------------------------------------------------------------------------------------------------#
# Functions
#----------------------------------------------------------------------------------------------------------------------#
def in_range(pd_data):
    burst_in_range = []
    for k in range(len(pd_data['Channel Label'])):
        if range_in_s[1] >= pd_data['Start timestamp [µs]'].iloc[k] / 1e6 >= range_in_s[0]:  # start within range
            if pd_data['End timestamp [µs]'].iloc[k] / 1e6 < range_in_s[1]:  # end within range
                burst_in_range.append([pd_data['Start timestamp [µs]'].iloc[k],
                                       pd_data['End timestamp [µs]'].iloc[k]])
            else:  # start in range/ end not in range
                burst_in_range.append([pd_data['Start timestamp [µs]'].iloc[k],
                                       range_in_s[1] * 1e6])
                break
        else:  # start not in range
            if range_in_s[1] >= pd_data['End timestamp [µs]'].iloc[k] / 1e6 >= range_in_s[0]:  # end in range
                burst_in_range.append([range_in_s[0] * 1e6, pd_data['End timestamp [µs]'].iloc[k]])
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

#----------------------------------------------------------------------------------------------------------------------#
# Variable setup
#----------------------------------------------------------------------------------------------------------------------#
range_in_s = (0, 300)  # range to be analysed in seconds [s]
allWells = ["A4", "A5", "B5", "C4", "C5", "D4", "D5"]
allElectrodes = [12, 13, 21, 22, 23, 24, 31, 32, 33, 34, 42, 43]
allDIVs = ['DIV04', 'DIV06', 'DIV08', 'DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24']

wellrainbow = np.array(["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"])

rootDir = os.path.dirname(os.path.abspath(__file__))

#----------------------------------------------------------------------------------------------------------------------#
# Spike dataframe construction
#----------------------------------------------------------------------------------------------------------------------#
# reading in spike data from the MCS software as csv into a pandas dataframe
spkdf = pd.read_csv(f'{rootDir}\\Hippocampus spk analysis.csv')
spkdf = spkdf.iloc[:, [1, 3, 6, 9]]  # Channel Label, Well Label, Experiment, Timestamp

uniNames = pd.unique(spkdf['Experiment'])
# work around for this experiment, further experiments need to be done with DIV easily slicable from the experiment col
spkdf['DIV'] = ['DIV04' if x == '20230516_Hippocampus_pMEA_001' else
                'DIV06' if x == '20230518_Hippocampus_pMEA_001' else
                'DIV08' if x == '20230520_Hippocampus_pMEA_001' else
                'DIV10' if x == '20230522_Hippocampus_pMEA_001' else
                'DIV12' if x == '20230524_Hippocampus_pMEA_001' else
                'DIV14' if x == '20230526_Hippocampus_pMEA_001' else
                'DIV18' if x == '20230530_Hippocampus_pMEA_001' else
                'DIV24' if x == '20230605_Hippocampus_pMEA_001' else 'nan' for x in spkdf["Experiment"]]

# Create a DataFrame with all possible combinations of dc, wc, and ec
combos = pd.DataFrame([(dc, wc, ec) for dc in allDIVs for wc in allWells for ec in allElectrodes],
                      columns=["DIV", "Well Label", "Channel Label"])
missing_combos = pd.merge(combos, spkdf, on=["DIV", "Well Label", "Channel Label"], how="left") # Merge to find missing combinations
missing_combos = missing_combos[missing_combos.isnull().any(axis=1)]
spkdf = pd.concat([spkdf, missing_combos], ignore_index=True, sort=False)

active = 0
FFdf = pd.DataFrame(columns=["Electrode", "Well", "DIV", "FF", "Active"])  # Firing rate dataframe
for d in pd.unique(spkdf["DIV"]):  # DIV seperating
    burst_array = []
    day = spkdf.loc[spkdf['DIV'] == d, :]
    for w in pd.unique(spkdf["Well Label"]):  # Well seperation
        well = day.loc[day['Well Label'] == w, :]

        for e in pd.unique(well["Channel Label"]):  # change to right object
            elecdata = well.loc[well['Channel Label'] == e, :]
            spks_in_range = elecdata.loc[elecdata['Timestamp [µs]'] > range_in_s[0] * 1e6, :]  # needs to be done in one line
            spks_in_range = spks_in_range.loc[spks_in_range['Timestamp [µs]'] < range_in_s[1] * 1e6, :]  # ^

            spk_int = np.diff(spks_in_range['Timestamp [µs]'])
            if len(spk_int) > 10:
                active += 1

            list_row = [e, w, d, len(spks_in_range) / 300, active]  # NaNs are throwing errors
            active = 0
            FFdf.loc[len(FFdf)] = list_row  # adding row to the dataframe for graphing

#----------------------------------------------------------------------------------------------------------------------#
# Plotting Spks
#----------------------------------------------------------------------------------------------------------------------#
# plt.title("Mean Firing Rate")
# plt.ylabel("Frequency [Hz]")
# c = 0
# for q in pd.unique(FFdf["Well"]):
#     x = FFdf.loc[FFdf["Well"] == q]
#     for _ in pd.unique(FFdf["Electrode"]):
#         y = x.loc[FFdf["Electrode"] == _]
#         plt.plot(y["DIV"], y["FF"], color=wellrainbow[c], alpha=0.3, lw=0.8)
#     c += 1
# plt.tight_layout()
# plt.show()
#----------------------------------------------------------------------------------------------------------------------#
# # Line plot
# plt.title("Active Channels")
# plt.ylabel("Active Channels")
# c = 0
# legend = []
# for well in pd.unique(FFdf["Well"]):
#     x = FFdf.loc[FFdf["Well"] == well]
#     z = []
#     for q in pd.unique(FFdf["DIV"]):
#         y = x.loc[x["DIV"] == q]
#         z.append(y["Active"].sum())
#         legend.append(q)
#     plt.plot(allDIVs, z, color=wellrainbow[c], alpha=0.5, lw=1.5, label=well, linestyle=':')
#     plt.legend()
#     c += 1
# plt.tight_layout()
# plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# Burst dataframe constrution
#----------------------------------------------------------------------------------------------------------------------#
# Reading in Burst data from the MCS software as csv into a pandas dataframe
bdf = pd.read_csv(f'{rootDir}\\Hippocampus bursts analysis.csv')
bdf = bdf.iloc[:, [1, 3, 6, 9, 10, 11, 12]]  # Channel Label, Well Label, Experiment, Start timestamp [µs], Duration [µs], Spike count, Spike Frequency [Hz]
bdf['End timestamp [µs]'] = bdf['Start timestamp [µs]'] + bdf['Duration [µs]']  # Adding End timestamp [µs]

# that's a neat piece of code
bdf['Experiment'] = ['DIV04' if x == '20230516_Hippocampus_pMEA_001' else
                       'DIV06' if x == '20230518_Hippocampus_pMEA_001' else
                       'DIV08' if x == '20230520_Hippocampus_pMEA_001' else
                       'DIV10' if x == '20230522_Hippocampus_pMEA_001' else
                       'DIV12' if x == '20230524_Hippocampus_pMEA_001' else
                       'DIV14' if x == '20230526_Hippocampus_pMEA_001' else
                       'DIV18' if x == '20230530_Hippocampus_pMEA_001' else
                       'DIV24' if x == '20230605_Hippocampus_pMEA_001' else 'nan' for x in bdf["Experiment"]]

# Create a DataFrame with all possible combinations of dc, wc, and ec
combos = pd.DataFrame([(dc, wc, ec) for dc in allDIVs for wc in allWells for ec in allElectrodes],
                      columns=["Experiment", "Well Label", "Channel Label"])
missing_combos = pd.merge(combos, bdf, on=["Experiment", "Well Label", "Channel Label"], how="left") # Merge to find missing combinations
missing_combos = missing_combos[missing_combos.isnull().any(axis=1)]  # Isolate missing combinations based on NaN
bdf = pd.concat([bdf, missing_combos], ignore_index=True, sort=False)



#----------------------------------------------------------------------------------------------------------------------#
# Analysis
#----------------------------------------------------------------------------------------------------------------------#
bdf['Duration [ms]'] = [x / 1e3 for x in bdf['Duration [µs]']]

# dbstdf = pd.DataFrame(columns=["Electrode", "Well", "DIV", "Hz"])
#
# for d in pd.unique(bstdf["Experiment"]):  # DIV seperating
#     burst_array = []
#     day = bstdf.loc[bstdf['Experiment'] == d, :]
#     for w in pd.unique(bstdf["Well Label"]):  # Well seperation
#         well = day.loc[day['Well Label'] == w, :]
#         for e in pd.unique(well["Channel Label"]):  # change to right object
#             elecdata = well.loc[well['Channel Label'] == e, :]  # select rows only with right electrode label
#
#             if mt.isnan(np.mean(elecdata["Spike Frequency [Hz]"])) :
#                 list_row = [e, w, d, 0]
#                 dbstdf.loc[len(dbstdf)] = list_row
#             else:
#                 hz = np.mean(elecdata["Spike Frequency [Hz]"])
#                 list_row = [e, w, d, hz]
#                 dbstdf.loc[len(dbstdf)] = list_row

#----------------------------------------------------------------------------------------------------------------------#
# Plotting Bursts
#----------------------------------------------------------------------------------------------------------------------#
# Bar plot with pandas
group = bdf.groupby(["Experiment"])[["Spike Count",
                                     "Duration [ms]",
                                     "Spike Frequency [Hz]"]].mean().plot(kind="bar", width=0.8, color=['#f09892',
                                                                                                        '#95f099',
                                                                                                        '#8db1eb'])
plt.title("Burst")  # average burst rate of all wells/day
plt.xlabel("")
plt.ylim(bottom=0)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
