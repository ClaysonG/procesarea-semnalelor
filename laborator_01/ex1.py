import numpy as np
import matplotlib.pyplot as plt

# simulez axa reala de timp
t = np.arange(0, 0.03, 0.0005)

# calculez semnalele continue x(t), y(t) si z(t)
x_t = np.cos(520 * np.pi * t + np.pi/3)
y_t = np.cos(280 * np.pi * t - np.pi/3)
z_t = np.cos(120 * np.pi * t + np.pi/3)

fig, axs = plt.subplots(3, 1, figsize=(8, 6))

# Afisez semnalele in subplot-uri separate
axs[0].plot(t, x_t)
axs[0].set_title('x(t)')

axs[1].plot(t, y_t)
axs[1].set_title('y(t)')

axs[2].plot(t, z_t)
axs[2].set_title('z(t)')

plt.tight_layout()
# plt.show()

fs = 200  # Frecventa de esantionare
n = np.arange(0, 0.03, 1/fs)

x_n = np.cos(520 * np.pi * n + np.pi/3)
y_n = np.cos(280 * np.pi * n - np.pi/3)
z_n = np.cos(120 * np.pi * n + np.pi/3)

fig, axs = plt.subplots(3, 1, figsize=(8, 6))

# Afisez semnalele esantionate peste cele continue in subplot-uri separate
axs[0].plot(t, x_t)
axs[0].stem(n, x_n)
axs[0].set_title('x[n]')

axs[1].plot(t, y_t)
axs[1].stem(n, y_n)
axs[1].set_title('y[n]')

axs[2].plot(t, z_t)
axs[2].stem(n, z_n)
axs[2].set_title('z[n]')

plt.tight_layout()
plt.show()
