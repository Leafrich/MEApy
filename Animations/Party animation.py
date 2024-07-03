import matplotlib.pyplot as plt
import matplotlib.animation as anime
import numpy as np
import random as rd

rgbList = []

for frame in range(12):
    rgbList.append(rd.sample(range(256), 3))
npRGB = np.array(rgbList)

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, x=None, y=None, colours=None):
        self.plot = None
        self.colours = colours
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots()
        self.ani = anime.FuncAnimation(self.fig, func=self.update, interval=10, frames=1000, init_func=self.setup_plot)
        self.ani.save("Party.mp4")

    def setup_plot(self):
        self.plot = plt.scatter(self.x, self.y, c=npRGB/255.0)
        self.ax.set_xlim(0.5, 4.5)
        self.ax.set_ylim(0.5, 4.5)
        self.ax.axes.invert_yaxis()
        self.ax.axis('off')
        return self.plot

    def update(self, frame):
        for el in range(len(self.colours)):
            self.colours[el] += rd.sample(range(-20,20), 3)
        self.colours[self.colours > 255] = 245
        self.colours[self.colours < 0] = 10
        self.plot.remove()
        self.plot = plt.scatter(self.x, self.y, c=self.colours/255.0, s=500)
        return self.plot,

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [6, 6]
    a = AnimatedScatter([1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4], [2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3], colours=npRGB)
    plt.show()
