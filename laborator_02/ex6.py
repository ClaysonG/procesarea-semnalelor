import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Parametrii semnalului
fs = 44100  # Frecventa de esantionare
T = 1 / fs  # Perioada de esantionare
t = np.arange(0, 1, T)  # Vector de timp pentru 1 secundă

# Generarea semnalului cu f = fs/2
f1 = fs / 2
signal1 = np.sin(2 * np.pi * f1 * t)

# Generarea semnalului cu f = fs/4
f2 = fs / 4
signal2 = np.sin(2 * np.pi * f2 * t)

# Generarea semnalului cu f = 0 Hz (semnal constant)
f3 = 0
signal3 = np.sin(2 * np.pi * f3 * t)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, signal1, label='Semnal cu f = fs/2')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.005)  # Afisez doar primele 5 ms

plt.subplot(3, 1, 2)
plt.plot(t, signal2, label='Semnal cu f = fs/4')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.005)  # Afisez doar primele 5 ms

plt.subplot(3, 1, 3)
plt.plot(t, signal3, label='Semnal cu f = 0 Hz')
plt.xlabel('Timp (secunde)')
plt.ylabel('Amplitudine')
plt.grid(True)
plt.legend()
plt.xlim(0, 0.005)  # Afisez doar primele 5 ms

plt.show()


print("Redarea semnalelor generate...")


def semnal_1():
    print("Semnal cu f = fs/2")
    sd.play(signal1, fs)
    sd.wait()


def semnal_2():
    print("Semnal cu f = fs/4")
    sd.play(signal2, fs)
    sd.wait()


def semnal_3():
    print("Semnal cu f = 0 Hz")
    sd.play(signal3, fs)
    sd.wait()


if __name__ == '__main__':
    # semnal_1()
    # semnal_2()
    # semnal_3()
    pass

# Observatie:
# Semnalul cu f = fs / 2 nu este afisat si nici redat (prea putine esantioane pentru a fi afisat? frecventa prea mare pentru place de sunet?)
# Semnalul cu f = fs / 4 este afisat si redat
# Semnalul cu f = 0 este afisat ca o dreapta si nu este redat
