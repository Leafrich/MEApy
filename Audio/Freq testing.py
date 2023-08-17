import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

dt = 0.0001  # time step

omegaT1 = 2 * np.pi * 3 * dt
omegaT2 = 2 * np.pi * 5 * dt
omegaT3 = 2 * np.pi * 1.083 * dt

rate = 10000

x = [10, 60, 30, 20]
tArray = np.array(np.repeat(x, [20000, 10000, 10000, 10000], axis=0))

x = np.arange(0, 5, 0.0001)
y = list()
freq = 0

print(len(x))
print(len(tArray))

for i in range(0, len(x)):
    c = np.sin(freq * 2 * np.pi)
    y.append(c)

    freq = freq + 2 * np.pi * tArray[i] * dt

    # # increment phase based on current frequency
    # if x[i] < 2:
    #     freq = freq + omegaT1
    # elif x[i] < 4:
    #     freq = freq + omegaT2
    # else:
    #     freq = freq + omegaT3


signalArray = np.array(y)
write('test.wav', rate, signalArray)

plt.plot(x, y)
plt.show()
