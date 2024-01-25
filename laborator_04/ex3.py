import numpy as np
import matplotlib.pyplot as plt


def genereaza_semnal(n: np.ndarray, f: float):
    """Generează un semnal sinusoidal."""
    return np.sin(2 * np.pi * f * n)


# Setarea parametrilor semnalului
frecventa = 4
frecventa_de_eșantionare = 15
n = np.linspace(0, 1, frecventa_de_eșantionare)
timp = np.linspace(0, 1, 10000)

# Crearea subplot-urilor
fig, axs = plt.subplots(4)

# Trasarea semnalului de bază
axs[0].plot(timp, genereaza_semnal(timp, frecventa))
axs[0].set_xlim([0, 1])

# Trasarea semnalului de bază cu puncte de eșantionare
axs[1].plot(timp, genereaza_semnal(timp, frecventa))
axs[1].stem(n, genereaza_semnal(n, frecventa))
axs[1].set_xlim([0, 1])

# Trasarea unui semnal cu frecvență crescută și puncte de eșantionare
axs[2].plot(timp, genereaza_semnal(timp, frecventa +
            frecventa_de_eșantionare), c='magenta')
axs[2].stem(n, genereaza_semnal(n, frecventa + frecventa_de_eșantionare))
axs[2].set_xlim([0, 1])

# Trasarea unui semnal cu o frecvență și mai mare și puncte de eșantionare
axs[3].plot(timp, genereaza_semnal(timp, frecventa +
            2 * frecventa_de_eșantionare), c='red')
axs[3].stem(n, genereaza_semnal(n, frecventa + 2 * frecventa_de_eșantionare))
axs[3].set_xlim([0, 1])

plt.tight_layout()

plt.show()
