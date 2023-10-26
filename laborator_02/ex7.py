import numpy as np
import matplotlib.pyplot as plt

# Parametrii semnalului
fs_initial = 1000  # Frecventa de esantionare initială (Hz)
T_initial = 1 / fs_initial  # Perioada de esantionare initială
# Vector de timp initial pentru 1 secundă
t_initial = np.arange(0, 1, T_initial)

# Generarea semnalului sinusoidal cu frecventa de esantionare initială
f = 100  # Frecventa semnalului sinusoidal (Hz)
signal = np.sin(2 * np.pi * f * t_initial)

# Decimarea semnalului la 1/4 din frecventa initială (pastrand doar al 4-lea element din vector)
signal_decimated = signal[::4]
t_decimated = t_initial[::4]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_initial, signal, label='Semnal initial')
plt.title('Semnal initial si semnal decimat')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.25)  # Afisez doar primele 250 ms

plt.subplot(2, 1, 2)
plt.plot(t_decimated, signal_decimated, label='Semnal decimat')
plt.xlabel('Timp (secunde)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.25)  # Afisez doar primele 250 ms

# Observatie: Semnalul decimat are o frecventa redusa și o perioada mai mare decat semnalul initial

# plt.show()

# Decimarea semnalului la 1/4 din frecvența initială (pornind de la al doilea element)
signal_decimated_b = signal[1::4]
t_decimated_b = t_initial[1::4]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_initial, signal, label='Semnal initial')
plt.title('Semnal initial si semnal decimat (pornind de la al doilea element)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.25)  # Afisez doar primele 250 ms

plt.subplot(2, 1, 2)
plt.plot(t_decimated_b, signal_decimated_b,
         label='Semnal decimat (pornind de la al doilea element)')
plt.xlabel('Timp (secunde)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.25)  # Afisez doar primele 250 ms

plt.show()

# Observatie: Semnalul decimat este identic cu cel anterior, dar incepe cu o perioada intarziata
