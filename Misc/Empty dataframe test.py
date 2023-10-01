import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)

wells = 7
allWells = ["A4", "A5", "B5", "C4", "C5", "D4", "D5"]
electrodes = 12
allElectrodes = [12, 13, 21, 22, 23, 24, 31, 32, 33, 34, 42, 43]
days = 8
allDIVs = ['DIV4', 'DIV6', 'DIV8', 'DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24']

rootDir = os.path.dirname(os.path.abspath(__file__))

bstdf = pd.read_csv(f'{rootDir}\\Hippocampus bursts analysis.csv')
bstdf = bstdf.iloc[:, [1, 3, 6, 9, 10, 11,
                       12]]  # Channel Label, Well Label, Experiment, Start timestamp [µs], Duration [µs], Spike count, Spike Frequency [Hz]
bstdf['End timestamp [µs]'] = bstdf['Start timestamp [µs]'] + bstdf['Duration [µs]']  # Adding End timestamp [µs]

# work around for this experiment, further experiments need to be done with DIV easily slicable from the experiment col
bstdf['DIV'] = ['DIV4' if x == '20230516_Hippocampus_pMEA_001' else
                'DIV6' if x == '20230518_Hippocampus_pMEA_001' else
                'DIV8' if x == '20230520_Hippocampus_pMEA_001' else
                'DIV10' if x == '20230522_Hippocampus_pMEA_001' else
                'DIV12' if x == '20230524_Hippocampus_pMEA_001' else
                'DIV14' if x == '20230526_Hippocampus_pMEA_001' else
                'DIV18' if x == '20230530_Hippocampus_pMEA_001' else
                'DIV24' if x == '20230605_Hippocampus_pMEA_001' else 'nan' for x in bstdf["Experiment"]]


# for dc in allDIVs:
#     if dc in bstdf["DIV"].unique(): continue
#     for wc in allWells:
#         for ec in allElectrodes:
#             list_row = [ec, wc, "Nan", 0, 0, 0, 0, 0, dc]
#             bstdf.loc[len(bstdf)] = list_row  # adding row to the dataframe for graphing
# for dc in allDIVs:
#     for wc in allWells:
#         if wc in bstdf["Well Label"].unique(): continue
#         for ec in allElectrodes:
#             list_row = [ec, wc, "Nan", 0, 0, 0, 0, 0, dc]
#             bstdf.loc[len(bstdf)] = list_row  # adding row to the dataframe for graphing
# for dc in allDIVs:
#     for wc in allWells:
#         for ec in allElectrodes:
#             if ec in bstdf["Channel Label"].unique(): continue
#             list_row = [ec, wc, "Nan", 0, 0, 0, 0, 0, dc]
#             bstdf.loc[len(bstdf)] = list_row  # adding row to the dataframe for graphing

div_set = set(bstdf["DIV"].unique())
well_set = set(bstdf["Well Label"].unique())
channel_set = set(bstdf["Channel Label"].unique())
for dc in allDIVs:
    for wc in allWells:
        for ec in allElectrodes:
            if dc not in div_set or wc not in well_set or ec not in channel_set:
                list_row = [ec, wc, "Nan", 0, 0, 0, 0, 0, dc]
                bstdf.loc[len(bstdf)] = list_row  # adding row to the dataframe


