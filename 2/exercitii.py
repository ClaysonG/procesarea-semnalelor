# Semnale periodice
# Un semnal se numeste periodic daca valorile sale se repeta intocmai dupa un
# interval de timp T, numit perioada (principala a) semnalului, x(t) = x(t + T).
# Se numeste frecventa fundamentala a unui semnal periodic valoarea f = 1/T.
# In practica nu se intalnesc semnale periodice pure, deoarece semnalele reale
# sunt, intr-o forma sau alta, afectate de zgomot, ce face ca valorile acestuia sa nu
# se repete identic. Cu toate ca acest concept este teoretic, analiza acestor tipuri
# de semnale este importanta in dezvoltarea unor metode ce se preteaza si pentru semnale aperiodice.
# Cel mai simplu semnal periodic il reprezinta semnalul sinusoidal, x(t) = Asin(2πft + φ) (1).
# In ecuatia de mai sus a este amplitudinea semnalului, anume valoarea maxima pe care o poate lua semnalul.
# Frecventa fundamentala, f, reprezinta numarul de oscilatii pe secunda (de cate ori intr-o secunda se repeta valoarea semnalului) si se masoara in Hz.
# Marimea φ reprezinta faza semnalului, se masoara in radiani si se refera la
# pozitia in cadrul perioadei in care se regaseste semnalul la t = 0.
# Ecuatia (1) se poate rescrie sub forma
# x(t) = Asin(ωt + φ), (2)
# unde ω reprezintaa frecventaa unghiulara si se masoara in radiani.
# Un semnal de tipul celui definit mai sus se numeste semnal sinusoidal.
# Asemanator cu acesta este si semnalul de tip cosinus, denumit de asemenea
# semnal sinusoidal. Diferenta intre cele doua este doar una de faza, datorita
# urmatoarei relatii cos(t) = sin(t + π/2).

# Cazul discret
# Alternativa discreta a semnalului (1) este
# x[n] = Asin(2πfnts + φ), (3)
# unde n reprezinta numarul esantionului, iar ts reprezinta perioada de esantionare.
# Frecventa de esantionare se defineste fs = 1/ts.
# Spre deosebire de semnalul sinusoidal continuu, care este periodic, cel discret
# este periodic doar in cazul in care raportul N = 2πk/ω este numar intreg, unde k ∈ Z.

# Ghid Python
# In plus fata de modulele folosite in laboratorul precedent, la acest laborator veti
# avea nevoie de scipy.io.wavfile, scipy.signal si sounddevice.
# Pentru a salva un semnal generat de voi in format audio puteti folosi urmatoarea secventa de cod:
# rate = int(10e5)
# scipy.io.wavfile.write(’nume.wav’, rate, signal).
# Daca doriti sa incarcati semnalul salvat anterior pentru a-l procesa in continuare, puteti folosi:
# rate, x = scipy.io.wavfile.read(’nume.wav’).
# Pentru a reda audio un semnal salvat intr-un numpy.array utilizati:
# sounddevice.play(myarray, fs),
# unde fs reprezinta frecventa de esantionare si o puteti seta fs = 44100.

import numpy as np
import matplotlib.pyplot as plt


def ex1():
    # Exercitiul 1
    # Generati un semnal sinusoidal de tip sinus, de amplitudine, frecventa si
    # faza aleasa de voi. Generati apoi un semnal de tip cosinus astfel incat pe
    # orizontul de timp ales, acesta sa fie identic cu semnalul sinus. Verificati
    # afisandu-le grafic in doua subplot-uri diferite.

    # Generez semnalul sinusoidal
    A = 1.0
    f = 10.0
    phi = 0
    t = np.linspace(0, 1, 1000)
    x1 = A * np.sin(2*np.pi*f*t + phi)

    # Generez semnalul cosinus
    x2 = A * np.cos(2*np.pi*f*t + phi - np.pi/2)

    # Afisez semnalele
    fig, axs = plt.subplots(2)
    axs[0].plot(t, x1)
    axs[0].set_title('Semnal sinusoidal')
    axs[1].plot(t, x2)
    axs[1].set_title('Semnal cosinus')
    plt.show()


def ex2():
    # Exercitiul 2
    # Generati un semnal sinusoidal de amplitudine unitara si frecventa aleasa
    # de voi. Incercati 4 valori diferite pentru faza. Afisati toate semnalele pe
    # acelasi grafic. Adaugati zgmot aleator sinusoidelor esantionate generate.
    # Noul semnal este x[n] + γz[n] astfel incat raportul semnal zgomot (Signal to Noise Ratio sau SNR) sa fie {0.1,1,10,100}.
    # Vectorul n este generat esantionand distributia Gaussiana
    # standard iar parametrul γ se calculeaza astfel incat sa avem valorile SNR dorite.

    # Generam semnalul sinusoidal
    A = 1.0
    f = 10.0
    phi = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4])
    t = np.linspace(0, 1, 1000)
    x = np.zeros((4, len(t)))
    for i in range(4):
        x[i, :] = A * np.sin(2*np.pi*f*t + phi[i])

    # Generam zgomotul
    n = np.random.randn(len(t))
    SNR = np.array([0.1, 1, 10, 100])
    gamma = np.sqrt(np.sum(x**2, axis=1) / (SNR * np.sum(n**2)))
    z = np.zeros((4, len(t)))
    for i in range(4):
        z[i, :] = gamma[i] * n

    # Adaugam zgomotul la semnalul original
    x_noisy = x + z

    # Afisam semnalele
    fig, ax = plt.subplots()
    for i in range(4):
        ax.plot(t, x_noisy[i, :], label=f'phi={phi[i]:.2f}')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # ex1()
    ex2()
