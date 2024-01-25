import numpy as np
import matplotlib.pyplot as plt


def genereaza_semnal(n, frecventa):
    """Generează un semnal sinusoidal."""
    return np.sin(2 * np.pi * frecventa * n)


# Definirea parametrilor semnalului
frecventa_de_baza = 10
frecventa_de_eșantionare = 15
valori_temporale = np.linspace(0, 1, 10000)
puncte_de_eșantionare = np.linspace(0, 1, frecventa_de_eșantionare + 1)

# Crearea subplot-urilor
fig, axs = plt.subplots(4, figsize=(10, 8))

# Trasarea semnalului de bază
axs[0].plot(valori_temporale, genereaza_semnal(
    valori_temporale, frecventa_de_baza))
axs[0].set_xlim([0, 1])
axs[0].set_title('Semnal Original')

# Trasarea semnalului de bază cu puncte de eșantionare
axs[1].plot(valori_temporale, genereaza_semnal(
    valori_temporale, frecventa_de_baza))
axs[1].stem(puncte_de_eșantionare, genereaza_semnal(
    puncte_de_eșantionare, frecventa_de_baza))
axs[1].set_xlim([0, 1])
axs[1].set_title('Semnal Original cu Puncte de Eșantionare')

# Trasarea unui semnal cu frecvență crescută și puncte de eșantionare
axs[2].plot(valori_temporale, genereaza_semnal(valori_temporale,
            frecventa_de_baza + frecventa_de_eșantionare), c='magenta')
axs[2].stem(puncte_de_eșantionare, genereaza_semnal(
    puncte_de_eșantionare, frecventa_de_baza + frecventa_de_eșantionare))
axs[2].set_xlim([0, 1])
axs[2].set_title('Semnal cu Frecvență Crescută')

# Trasarea unui semnal cu o frecvență și mai mare și puncte de eșantionare
axs[3].plot(valori_temporale, genereaza_semnal(valori_temporale,
            frecventa_de_baza + 2 * frecventa_de_eșantionare), c='red')
axs[3].stem(puncte_de_eșantionare, genereaza_semnal(
    puncte_de_eșantionare, frecventa_de_baza + 2 * frecventa_de_eșantionare))
axs[3].set_xlim([0, 1])
axs[3].set_title('Semnal cu Frecvență Încă Mai Mare')

fig.tight_layout()

plt.show()
