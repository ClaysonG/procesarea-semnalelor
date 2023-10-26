import numpy as np
import matplotlib.pyplot as plt

N = 8

# Creare matrice Fourier
F = np.zeros((N, N), dtype=np.complex128)
for n in range(N):
    for k in range(N):
        F[n, k] = np.exp(-2j * np.pi * n * k / N) / np.sqrt(N)

# Verific daca matricea Fourier este ortogonala
is_orthogonal = np.allclose(np.eye(N), F @ F.conj().T)
print(f"Matricea Fourier este ortogonala: {is_orthogonal}")

# Verific daca matricea este complexa
is_complex = np.allclose(F, F.conj().T)
print(f"Matricea Fourier este complexa: {is_complex}")

# Verific daca matricea este unitara
is_unitary = is_complex and is_orthogonal
print(f"Matricea Fourier este unitara: {is_unitary}")

# Afisare grafice pentru fiecare componenta de frecventa
plt.figure(figsize=(12, 8))

for k in range(N):
    plt.subplot(N, 2, 2 * k + 1)
    plt.plot(np.real(F[:, k]))
    plt.title(f"Partea Reala a Componentei {k+1}")
    plt.subplot(N, 2, 2 * k + 2)
    plt.plot(np.imag(F[:, k]))
    plt.title(f"Partea Imaginara a Componentei {k+1}")
    plt.tight_layout()

plt.show()
