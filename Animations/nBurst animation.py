import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import os

def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
    return A,B,C

pd.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)

root = os.path.dirname(os.path.abspath(os.path.join(__file__ ,"..")))
df = pd.read_excel(f'{root}\\Data\\nBurst animation.xlsx')
df = df.astype({"Channel Label": str})
df[["Row", "Col"]] = df["Channel Label"].apply(list).tolist()
df = df.astype({"Row": int, "Col": int})

tpf = 1000

frames = ((df.max(axis="rows", numeric_only=True)["Start timestamp [µs]"] + df.max(axis="rows", numeric_only=True)["Duration [µs]"])
         - df.min(axis="rows", numeric_only=True)["Start timestamp [µs]"]) / tpf

starting = df.min(axis="rows", numeric_only=True)["Start timestamp [µs]"]

xcurve = [(frames/2)-frames, frames/2]

anidf = pd.DataFrame(columns=["Row", "Col", "Colour", "Frame"])  # Firing rate dataframe
for electrode in df["Channel Label"].unique():
    x1, y1 = [int((df[df["Channel Label"] == electrode]["Start timestamp [µs]"]) - starting) / tpf, 1]
    x3, y3 = [int((df[df["Channel Label"] == electrode]["Start timestamp [µs]"] +
              (df[df["Channel Label"] == electrode]["Duration [µs]"])) - starting) / tpf, 1]
    x2, y2 = [(x1 + x3) / 2, 0]
    frame = 0
    while frame < int(frames):
        tmp = df[df["Channel Label"] == electrode]
        tmp = tmp.reset_index()
        row = list(tmp.loc[0, ["Row", "Col"]])
        if frame <= x1 or frame >= x3:
            list_row = [row[0], row[1], [1, 1, 1], frame]
            anidf.loc[len(anidf)] = list_row
            frame += 1
        else:
            a, b, c = calc_parabola_vertex(x1, y1, x2, y2, x3, y3)
            x_pos=np.arange(x1, x3, 1)
            y_pos=[]
            print(f'Start for electrode {electrode}, frame: {frame}, len: {len(x_pos)}')
            for x in range(len(x_pos)):
                x_val=x_pos[x]
                y=(a*(x_val**2))+(b*x_val)+c
                if y > 1:
                    y = 1
                if y < 0:
                    y = 0
                list_row = [row[0], row[1], [y, y, y], frame]
                anidf.loc[len(anidf)] = list_row
                frame += 1

#
# for el in test:
#     x = range(int(frames))
#     tmp = anidf.loc[(anidf['Row'] == el[0]) & (anidf['Col'] == el[1])]
#     tmp = list(tmp["Colour"])
#     plt.plot(x, tmp)
# plt.show()

# for i in range(12):
# 	row = list(df.loc[i, ["Row", "Col"]])
# 	tdf = list(df.loc[i, ["Row", "Col", "Start timestamp [µs]"]])
# 	fr = 0
# 	print(f'{round((i / 12) * 100)}%')
# 	for frame in range(int(frames)):
# 		startFrame = df.min(axis="rows", numeric_only=True)["Start timestamp [µs]"]
# 		beginFrame = min(list(df.loc[i, ["Row", "Col"]]))
# 		endFrame = max(list(df.loc[i, ["Row", "Col"]]))
# 		print(startFrame, beginFrame)
# 		if startFrame < beginFrame or beginFrame + frame > endFrame:  # is this even right?
# 			Colour = 0
# 			list_row = [row[0], row[1], [Colour, Colour, Colour], fr]
# 		else:
# 			for j in np.arange(xcurve[0], xcurve[1], 1):
# 				Colour = (pow(0.07 * j, 2)) / 255
# 				list_row = [row[0], row[1], [Colour, Colour, Colour], fr]
# 				anidf.loc[len(anidf)] = list_row
# 				fr += 1
# 				frame += 1

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, data=None, frames=None):
        self.electrodes = [[1,1,2,2,2,2,3,3,3,3,4,4], [2,3,1,2,3,4,1,2,3,4,2,3]]
        self.plot = None
        self.df = data
        self.frames = frames
        self.fig, self.ax = plt.subplots()
        self.ani = animation.FuncAnimation(self.fig, func=self.update, interval=10, frames=self.frames, init_func=self.setup_plot)
        self.ani.save("nBurst 25fps.mp4", fps=10)

    def setup_plot(self):
        data = self.df[self.df["Frame"] == 0]
        # self.plot = plt.scatter(x=self.electrodes[0], y=self.electrodes[1], facecolors='none', edgecolors='#000000', s=500, linewidths=3)
        self.plot = plt.scatter(x=self.electrodes[0], y=self.electrodes[1], facecolors='#FFFFFF', edgecolors='#FFFFFF')
        self.ax.set_xlim(0.5, 4.5)
        self.ax.set_ylim(0.5, 4.5)
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.axes.invert_yaxis()
        self.ax.axis('off')
        return self.plot

    def update(self, frame):
        data = self.df[self.df["Frame"] == frame]
        self.plot.remove()
        # self.plot = plt.scatter(x=self.electrodes[0], y=self.electrodes[1], facecolors='none', edgecolors='#000000', s=500, linewidths=3)
        self.plot = plt.scatter(x=data["Row"], y=data["Col"], c=data["Colour"], s=500)
        return self.plot,

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [6, 6]
    a = AnimatedScatter(data=anidf, frames=int(frames))
    plt.show()
