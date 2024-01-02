import numpy as np
import matplotlib.pyplot as plt
import time


def signal(n: np.ndarray, f: float):
    return np.sin(2 * np.pi * f * n)


f = 4
fs = 15
n = np.linspace(0, 1, fs)
time = np.linspace(0, 1, 10000)

fig, axs = plt.subplots(4)
axs[0].plot(time, signal(time, f))
axs[0].set_xlim([0, 1])

axs[1].plot(time, signal(time, f))
axs[1].stem(n, signal(n, f))
axs[1].set_xlim([0, 1])

axs[2].plot(time, signal(time, f + fs), c='magenta')
axs[2].stem(n, signal(n, f + fs))
axs[2].set_xlim([0, 1])

axs[3].plot(time, signal(time, f + 2 * fs), c='red')
axs[3].stem(n, signal(n, f + 2 * fs))
axs[3].set_xlim([0, 1])

plt.tight_layout()

plt.show()
plt.close(fig)
