import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
from scipy.io import wavfile


def listen_and_save(signal, fs, title):
    sd.play(signal, fs)
    sd.wait()
    # write to current path
    wavfile.write(f'{title}.wav', fs, signal)


# Parametrii semnalului
fs = 44100  # Frecventa de esantionare pentru audio (de obicei 44100 Hz)
T = 1 / fs  # Perioada de esantionare
t = np.arange(0, 1, T)  # Vector de timp pentru 1 secundă


def a():
    A = 1.0  # Amplitudine
    f = 400.0  # Frecventa (Hz)

    signal = A * np.sin(2 * np.pi * f * t)

    listen_and_save(signal, fs, 'semnal_a')


def b():
    A = 1.0  # Amplitudine
    f = 800.0  # Frecventa (Hz)

    signal = A * np.sin(2 * np.pi * f * t)

    listen_and_save(signal, fs, 'semnal_b')


def c():
    f = 240.0  # Frecventa (Hz)

    signal = np.mod(f * t, 1)

    listen_and_save(signal, fs, 'semnal_c')


def d():
    A = 1.0  # Amplitudine
    f = 300.0  # Frecventa (Hz)

    signal = np.sign(A * np.sin(2 * np.pi * f * t))

    listen_and_save(signal, fs, 'semnal_d')


def read_file(filename):
    content = scipy.io.wavfile.read(filename)
    print(content)


if __name__ == '__main__':
    # a()
    # b()
    # c()
    # d()
    read_file('semnal_a.wav')
