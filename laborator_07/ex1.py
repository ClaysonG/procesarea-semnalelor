import numpy as np
import matplotlib.pyplot as plt

# Exercitiul 1 - 1


def func1(n1, n2):
    return np.sin(2 * np.pi * n1 + 3 * np.pi * n2)


dim = 512
n1, n2 = np.meshgrid(range(dim), range(dim))

X = func1(n1, n2)
Y = np.fft.fftshift(np.fft.fft2(X))
freq_db = 20*np.log10(abs(Y))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Functia 1')

img = axs[0].imshow(X)
axs[0].set_title("Semnalul")
fig.colorbar(img, ax=axs[0])

img = axs[1].imshow(freq_db)
axs[1].set_title("Spectrul semnalului")
fig.colorbar(img, ax=axs[1])

plt.tight_layout()

plt.show()

# Exercitiul 1 - 2


def func2(n1, n2):
    return np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)


dim = 512
n1, n2 = np.meshgrid(range(dim), range(dim))

X = func2(n1, n2)
Y = np.fft.fftshift(np.fft.fft2(X))
freq_db = 20*np.log10(abs(Y + 1e-20))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Functia 2')

img = axs[0].imshow(X)
axs[0].set_title("Semnalul")
fig.colorbar(img, ax=axs[0])

img = axs[1].imshow(freq_db)
axs[1].set_title("Spectrul semnalului")
fig.colorbar(img, ax=axs[1])

plt.tight_layout()

plt.show()

# Exercitiul 1 - 3
dim = 30

Y = np.zeros((dim, dim), dtype=complex)
Y[0, 5] = 1
Y[0, dim - 5] = 1

X = np.real(np.fft.ifft2(Y))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Functia 3')

img = axs[0].imshow(X)
axs[0].set_title("Semnalul")
fig.colorbar(img, ax=axs[0])

img = axs[1].imshow(np.abs(np.fft.fftshift(Y)) + 1e-15)
axs[1].set_title("Spectrul semnalului")
fig.colorbar(img, ax=axs[1])

plt.tight_layout()

plt.show()

# Exercitiul 1 - 4
dim = 30

Y = np.zeros((dim, dim), dtype=complex)
Y[5, 0] = 1
Y[dim - 5, 0] = 1

X = np.real(np.fft.ifft2(Y))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Functia 4')

img = axs[0].imshow(X)
axs[0].set_title("Semnalul")
fig.colorbar(img, ax=axs[0])

img = axs[1].imshow(np.abs(np.fft.fftshift(Y)) + 1e-15)
axs[1].set_title("Spectrul semnalului")
fig.colorbar(img, ax=axs[1])

plt.tight_layout()

plt.show()

# Exercitiul 1 - 5
dim = 30

Y = np.zeros((dim, dim), dtype=complex)
Y[5, 5] = 1
Y[dim - 5, dim - 5] = 1

X = np.real(np.fft.ifft2(Y))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Functia 5')

img = axs[0].imshow(X)
axs[0].set_title("Semnalul")
fig.colorbar(img, ax=axs[0])

img = axs[1].imshow(np.abs(np.fft.fftshift(Y)) + 1e-15)
axs[1].set_title("Spectrul semnalului")
fig.colorbar(img, ax=axs[1])

plt.tight_layout()

plt.show()
