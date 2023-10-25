import numpy as np
import matplotlib.pyplot as plt


def a():
    # (a) Generare semnal sinusoidal
    fs_a = 400  # Frecventa semnalului (Hz)
    n_a = 1600  # Numarul de esantioane

    t_a = np.arange(0, n_a / fs_a, 1 / n_a)
    signal_a = np.sin(2 * np.pi * fs_a * t_a)

    plt.plot(t_a, signal_a)
    plt.title('Semnal sinusoidal de 400 Hz cu 1600 de esantioane')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.xlim(0, 0.05)  # Limitez afisarea la primele 0.05 secunde
    plt.show()


def b():
    # (b) Generare semnal sinusoidal
    fs_b = 800  # Frecventa semnalului (Hz)
    duration_b = 3  # Durata semnalului (secunde)
    n_b = duration_b * fs_b  # Numarul de esantioane

    t_b = np.arange(0, duration_b, 1 / n_b)
    signal_b = np.sin(2 * np.pi * fs_b * t_b)

    plt.plot(t_b, signal_b)
    plt.title('Semnal sinusoidal de 800 Hz cu durata de 3 secunde')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.xlim(0, 0.05)  # Limitez afisarea la primele 0.05 secunde
    plt.show()


def c():
    # (c) Generare semnal sawtooth
    fs_c = 240  # Frecventa semnalului (Hz)
    duration_c = 3  # Durata semnalului (secunde)
    n_c = int(fs_c * duration_c) * 100  # Numarul de esantioane

    t_c = np.linspace(0, duration_c, n_c, endpoint=False)
    signal_c = np.mod(t_c * fs_c, 1)

    plt.plot(t_c, signal_c)
    plt.title('Semnal sawtooth de 240 Hz')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.xlim(0, 0.05)  # Limitez afisarea la primele 0.05 secunde
    plt.show()


def d():
    # (d) Generare semnal square
    fs_d = 300  # Frecventa semnalului (Hz)
    duration_d = 3  # Durata semnalului (secunde)
    n_d = int(fs_d * duration_d) * 100  # Numarul de esantioane

    t_d = np.linspace(0, duration_d, n_d, endpoint=False)

    signal_d = np.sign(np.sin(2 * np.pi * fs_d * t_d))

    plt.plot(t_d, signal_d)
    plt.title('Semnal square de 300 Hz')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.xlim(0, 0.025)  # Limitez afisarea la primele 0.025 secunde
    plt.show()


def e():
    # (e) Generare semnal 2D aleator
    x, y = 128, 128  # Dimensiunea matricei
    signal_e = np.random.rand(x, y)

    # Afișare semnal
    plt.imshow(signal_e)
    plt.title('Semnal 2D aleator (128x128)')
    plt.show()


def f():
    # (f) Generare semnal 2D personalizat
    x, y = 128, 128  # Dimensiunea matricei
    signal_f = np.zeros((x, y))  # Inițializare cu o matrice de zero-uri

    # Dungi orizontale
    signal_f[0:32, :] = 1
    signal_f[32:64, :] = 0.75
    signal_f[64:96, :] = 0.5
    signal_f[96:128, :] = 0.25

    # Dungi verticale
    signal_f[:, 0:16] = 0.2
    signal_f[:, 32:48] = 0.4
    signal_f[:, 64:80] = 0.6
    signal_f[:, 96:112] = 0.8

    # Afișare semnal
    plt.imshow(signal_f)
    plt.title('Semnal 2D personalizat (128x128)')
    plt.show()


if __name__ == '__main__':
    # a()
    # b()
    # c()
    # d()
    # e()
    f()
