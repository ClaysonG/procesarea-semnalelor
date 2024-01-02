import numpy as np
import matplotlib.pyplot as plt


def fereastra_dreptunghiulara(N: int):
    return np.ones(N)


def fereastra_hanning(N: int):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))


def sinusoida(A: float, f: float, t: int):
    return A * np.sin(2 * np.pi * f * t)


f = 100
fs = 1000
N = 200
t = np.arange(N) / fs
print(np.arange(10) / fs)
semnal = sinusoida(1, f, t)  # type: ignore

semnal_dreptunghiular = semnal * fereastra_dreptunghiulara(N)
semnal_hanning = semnal * fereastra_hanning(N)

fig, axs = plt.subplots(3, 1, figsize=(10, 7))
axs[0].plot(semnal)
axs[0].set_title('Semnalul original')

axs[1].plot(semnal_dreptunghiular, color='green')
axs[1].set_title('Semnalul cu fereastra dreptunghiulara')

axs[2].plot(semnal_hanning, color='orange')
axs[2].set_title('Semnalul cu fereastra hanning')

for ax in axs:
    ax.grid(True)

plt.tight_layout()

plt.show()

plt.figure(figsize=(10, 6))

plt.plot(t, semnal_dreptunghiular,
         label='Sinusoida cu fereastra dreptunghiulara', color='green')
plt.plot(t, semnal_hanning, label='Sinusoida cu fereastra Hanning', color='orange')
plt.title('Comparare ferestre')
plt.legend()

plt.tight_layout()

plt.show()
