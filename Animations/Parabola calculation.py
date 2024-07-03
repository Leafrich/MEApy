import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    '''
    Adapted and modifed to get the unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''

    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

    return A,B,C

pd.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)

root = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(f'{root}\\nBurst animation.xlsx')
df = df.astype({"Channel Label": str})
df[["Row", "Col"]] = df["Channel Label"].apply(list).tolist()
df = df.astype({"Row": int, "Col": int})
starting = df.min(axis="rows", numeric_only=True)["Start timestamp [µs]"]
print(df[df["Channel Label"] == "33"]["Start timestamp [µs]"])

x1,y1=[int((df[df["Channel Label"] == "33"]["Start timestamp [µs]"])-int(starting))/1000, 0]
x3,y3=[int((df[df["Channel Label"] == "33"]["Start timestamp [µs]"] + (df[df["Channel Label"] == "33"]["Duration [µs]"]))-int(starting))/1000, 0]
x2,y2=[(x1+x3)/2, 255]
print(x1, y1, x2, y2)

#Calculate the unknowns of the equation y=ax^2+bx+c
a,b,c=calc_parabola_vertex(x1, y1, x2, y2, x3, y3)

#Define x range for which to calc parabola
starting = df.min(axis="rows", numeric_only=True)["Start timestamp [µs]"] / 1000
x_pos=np.arange(x1, x3, 1)
print(x_pos)
y_pos=[]

#Calculate y values
for x in range(len(x_pos)):
    x_val=x_pos[x]
    y=(a*(x_val**2))+(b*x_val)+c
    y_pos.append(y)

# Plot the parabola (+ the known points)

plt.plot(x_pos, y_pos, linestyle='-.', color='black') # parabola line
plt.scatter(x_pos, y_pos, color='gray') # parabola points
plt.scatter(x1,y1,color='r',marker="D",s=50) # 1st known xy
plt.scatter(x2,y2,color='g',marker="D",s=50) # 2nd known xy
plt.scatter(x3,y3,color='k',marker="D",s=50) # 3rd known xy
plt.show()