import numpy as np
import matplotlib.pyplot as plt

# Generare vector aleator
N = 100
x = np.random.rand(N)

# Salvare pentru comparare ulterioara
x_initial = x.copy()

# Configurare subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 6))

# Calculul iterat si afisare in subplots separate
for i in range(3):
    row = i // 2
    col = i % 2

    x = np.convolve(x, x, mode='same')

    # Normalizare
    x /= np.linalg.norm(x)

    axs[row, col].plot(x, label=f'Iteratia {i + 1}')
    axs[row, col].set_title(f'Iteratia {i + 1}')

# Afisare graficul initial pentru comparatie
axs[1, 1].plot(x_initial, label='Initial', linestyle='--')
axs[1, 1].set_title('Initial')

# Afisare legende
for ax in axs.flat:
    ax.legend()

# Ajustare aspect pentru a evita suprapunerea textului
plt.tight_layout()
plt.show()

# Observatie: cu fiecare iteratie apare o gaussiana din ce in ce mai "perfecta"
