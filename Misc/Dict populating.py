import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)

rootDir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
df = pd.read_csv(f'{rootDir}\\Data\\Hippocampus bursts analysis.csv')
df = df.iloc[:, [1, 3, 6, 9, 10, 11, 12]]
df['End timestamp [µs]'] = df['Start timestamp [µs]'] + df['Duration [µs]']

uniNames = pd.unique(df['Experiment'])
DIVarr = ['DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24']
Dict = {}

df['ID'] = df['Well Label'].astype(str) + df['Channel Label'].astype(str)

for _ in range(len(uniNames)):
    Dict[DIVarr[_]] = df.loc[df['Experiment'] == uniNames[_], :]

print(Dict)
