import numpy as np
import matplotlib.pyplot as plt

# Parametrii semnalului
fs = 44100  # Frecventa de eșantionare
T = 1 / fs  # Perioada de eșantionare
t = np.arange(0, 1, T)  # Vector de timp pentru 1 secundă

# Semnal sinusoidal
A_sine = 1.0  # Amplitudine
f_sine = 1000.0  # Frecventa (Hz)
sine_signal = A_sine * np.sin(2 * np.pi * f_sine * t)

# Semnal sawtooth
A_sawtooth = 0.5  # Amplitudine
f_sawtooth = 500.0  # Frecventa (Hz)
sawtooth_signal = np.mod(f_sawtooth * t, 1)

# Suma celor doua semnale
sum_signal = sine_signal + sawtooth_signal

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, sine_signal, label='Semnal sinusoidal')
plt.title('Semnale diferite si suma lor')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.005)  # Afisez doar primele 5 ms

plt.subplot(3, 1, 2)
plt.plot(t, sawtooth_signal, label='Semnal sawtooth')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.005)  # Afisez doar primele 5 ms

plt.subplot(3, 1, 3)
plt.plot(t, sum_signal, label='Suma semnalelor')
plt.xlabel('Timp (secunde)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.005)  # Afisez doar primele 5 ms

plt.show()
