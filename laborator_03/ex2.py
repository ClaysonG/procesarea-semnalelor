import numpy as np
import matplotlib.pyplot as plt

f = 10  # Frecventa semnalului
fs = 1000  # Frecventa de esantionare

t = np.linspace(0, 1, fs)   # Timpul semnalului
x = np.sin(2 * np.pi * f * t)

# Infasurarea semnalului x(t) pe cercul unitate
y = x * np.exp(-2j * np.pi * t)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(t, x)
axs[0].set_title("Semnalul x(t)")
axs[0].set_xlabel("Timp")
axs[0].set_ylabel("Amplitudine")
axs[0].grid()

axs[1].plot(y.real, y.imag, c='green')
axs[1].set_title("Reprezentarea semnalului in planul complex (infasurarea)")
axs[1].set_xlabel("Real")
axs[1].set_ylabel("Imaginar")
axs[1].grid()

plt.tight_layout()

w = [1, 2, 3, 10]  # Frecventele de infasurare
z = [x * np.exp(-2j * np.pi * fw * t) for fw in w]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

colors = [np.abs(zi) for zi in z]

for i in range(len(w)):
    row, col = i // 2, i % 2
    axs[row, col].scatter(z[i].real, z[i].imag,
                          label=f'w = {w[i]}', c=colors[i])
    axs[row, col].legend()
    axs[row, col].set_title(f'Reprezentare pentru w = {w[i]}')
    axs[row, col].set_xlabel("Real")
    axs[row, col].set_ylabel("Imaginar")
    axs[row, col].grid()

plt.tight_layout()
plt.show()
