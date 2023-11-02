import numpy as np
import time
import matplotlib.pyplot as plt

# Dimensiunile vectorilor N
Ns = [128, 256, 512, 1024, 2048, 4096, 8192]

# Listele pentru stocarea timpilor de executie
t_manual = []
t_numpy = []

for N in Ns:
    # Generare vector de dimensiune N
    x = np.random.random(N)

    # Implementarea "de mana"
    start_time = time.time()
    X_manual = np.zeros((N, N), dtype=np.complex128)
    for n in range(N):
        for k in range(N):
            X_manual[n, k] = np.exp(-2j * np.pi * n * k / N) / np.sqrt(N)
    end_time = time.time()
    t_manual.append(end_time - start_time)

    # Utilizarea functiei numpy.fft
    start_time = time.time()
    X_numpy = np.fft.fft(x)
    end_time = time.time()
    t_numpy.append(end_time - start_time)

# Afisare grafic
plt.figure(figsize=(10, 6))
plt.plot(Ns, t_manual, label="Implementare \"de mana\"")
plt.plot(Ns, t_numpy, label="numpy.fft")
plt.yscale('log')  # Scara logaritmica pe axa Oy
plt.xlabel('Dimensiunea vectorului N')
plt.ylabel('Timp de executie (s)')
plt.legend()
plt.title('Comparatie timp de executie transformata Fourier')
plt.show()
