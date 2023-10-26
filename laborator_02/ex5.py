import numpy as np
import sounddevice as sd

# Parametrii semnalului
fs = 44100  # Frecventa de eșantionare
T = 1 / fs  # Perioada de esantionare
t = np.arange(0, 1, T)  # Vector de timp pentru 1 secundă

A = 1.0  # Amplitudine
f1 = 1000.0  # Frecvența primului semnal (Hz)
f2 = 2000.0  # Frecvența celui de-al doilea semnal (Hz)

# Generare semnal cu prima frecvență
signal1 = A * np.sin(2 * np.pi * f1 * t)

# Generare semnal cu a doua frecvență
signal2 = A * np.sin(2 * np.pi * f2 * t)

# Concatenez cele doua semnale
concatenated_signal = np.concatenate((signal1, signal2))

# Redare audio a semnalului rezultat
print("Redarea semnalului rezultat...")
sd.play(concatenated_signal, fs)
sd.wait()

# Observatie: Se aude tranzitia de la o frecventa la alta.
