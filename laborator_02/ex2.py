import numpy as np
import matplotlib.pyplot as plt

# Parametrii semnalului
fs = 10000  # Frecventa de esantionare (Hz)
T = 1 / fs  # Perioada de esantionare
t = np.arange(0, 1, T)  # Vector de timp pentru 1 secundă

# Amplitudinea si frecventa semnalului sinusoidal
A = 1.0  # Amplitudine
f = 1000.0  # Frecventa (Hz)

# Faze pentru cele 4 semnale
phases = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Semnalul sinusoidal original
signal = A * np.sin(2 * np.pi * f * t)

# Generare zgomot aleator
noise = np.random.normal(0, 1, len(t))

# Calcularea parametrului γ pentru fiecare SNR
SNR_values = [0.1, 1, 10, 100]
gamma_values = [np.sqrt(np.linalg.norm(signal)**2 /
                        (SNR * np.linalg.norm(noise)**2)) for SNR in SNR_values]

plt.figure(figsize=(10, 6))

plt.plot(t, signal, label='Semnal original', linewidth=2)

# Afisarea semnalelor cu zgomot
for i, gamma in enumerate(gamma_values):
    noisy_signal = signal + gamma * noise

    plt.plot(t, noisy_signal, label=f'Semnal cu SNR = {SNR_values[i]}')

plt.title('Semnale sinusoidale cu zgomot')
plt.xlabel('Timp (secunde)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.005)  # Afisez doar primele 5 ms
plt.show()
