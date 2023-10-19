import matplotlib.pyplot as plt
import numpy as np

# Exercitiul 1
def ex1():
    # Fie semnalele continue x(t) = cos(520πt + π/3), y(t) = cos(280πt −π/3) si z(t) = cos(120πt + π/3).

    # (a) Simulati axa reala de timp printr-un sir de numere suficient de apropiate, spre exemplu [0 : 0.0005 : 0.03].

    # Definim axa reala de timp
    t = np.arange(0, 0.03, 0.0005)

    # (b) Construiti semnalele x(t), y(t) si z(t) si afisati-le grafic, in cate un subplot.

    # Definim semnalele continue
    x = np.cos(520 * np.pi * t + np.pi / 3)
    y = np.cos(280 * np.pi * t - np.pi / 3)
    z = np.cos(120 * np.pi * t + np.pi / 3)

    # Afisam semnalele continue
    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.ylabel('x(t)')

    plt.subplot(3, 1, 2)
    plt.plot(t, y)
    plt.ylabel('y(t)')

    plt.subplot(3, 1, 3)
    plt.plot(t, z)
    plt.ylabel('z(t)')

    plt.xlabel('t') 
    # plt.show()

    # (c) Esantionati semnalele cu o frecventa de 200 Hz pentru a obtine x[n], y[n] si z[n] si afisati-le grafic, in cate un subplot.

    # Definim axa discreta de timp
    tn = np.arange(0, 0.03, 1 / 200)

    # Definim semnalele discrete
    xn = np.cos(520 * np.pi * tn + np.pi / 3)
    yn = np.cos(280 * np.pi * tn - np.pi / 3)
    zn = np.cos(120 * np.pi * tn + np.pi / 3)

    # Afisam semnalele continue
    plt.figure(2)

    plt.subplot(3, 1, 1)
    plt.plot(t, x)

    plt.subplot(3, 1, 2)
    plt.plot(t, y)

    plt.subplot(3, 1, 3)
    plt.plot(t, z)

    # Afisam semnalele discrete
    plt.figure(2)

    plt.subplot(3, 1, 1)
    plt.stem(tn, xn)
    plt.ylabel('x[n]')

    plt.subplot(3, 1, 2)
    plt.stem(tn, yn)
    plt.ylabel('y[n]')

    plt.subplot(3, 1, 3)
    plt.stem(tn, zn)
    plt.ylabel('z[n]')

    plt.xlabel('n')
    plt.show()

# Exercitiul 2
def ex2():
    # Generati urmatoarele semnale si afisati-le grafic, fiecare intr-un plot

    # (a) Un semnal sinusoidal de frecventa 400 Hz, care sa contina 1600 de esantioane.

    # Definim axa discreta de timp
    t = np.arange(0, 4, 1 / 1600)

    # Definim semnalul sinusoidal
    x = np.sin(2 * np.pi * 400 * t)

    # Afisam semnalul sinusoidal
    plt.figure(1)
    plt.plot(t, x)

    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.show()

if __name__ == '__main__':
    ex1()
    # ex2()
