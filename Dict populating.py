import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)

rootDir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f'{rootDir}\\Hippocampus bursts analysis.csv')
df = df.iloc[:, [1, 3, 6, 9, 10, 11, 12]]
df['End timestamp [µs]'] = df['Start timestamp [µs]'] + df['Duration [µs]']

uniNames = pd.unique(df['Experiment'])
DIVarr = ['DIV10', 'DIV12', 'DIV14', 'DIV18', 'DIV24']

for _ in range(len(DIVarr)):

    for i in range(len(uniNames)):




