import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.linspace(0, 1, N)

f = [20, 30, 50, 70]    # Frecventele semnalelor (4 componente)

semnal = np.zeros_like(t)   # Semnalul compus
for fi in f:
    semnal += np.sin(2 * np.pi * fi * t)

semnal_dft = np.zeros(N, dtype=complex)  # Transformata Fourier
for k in range(N):
    for n in range(N):
        semnal_dft[k] += semnal[n] * np.e ** (-2j * np.pi * k * n / N)

semnal_dft = np.abs(semnal_dft)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(t, semnal)
axs[0].set_title("Semnalul x(t)")
axs[0].set_xlabel("Timp")
axs[0].set_ylabel("Amplitudine")
axs[0].grid()

axs[1].stem(np.linspace(0, 100, 100), semnal_dft[:100])
axs[1].set_title("Transformata Fourier a semnalului")
axs[1].set_xlabel("Frecventa")
axs[1].set_ylabel("Amplitudine")
axs[1].grid()

plt.tight_layout()
plt.show()
