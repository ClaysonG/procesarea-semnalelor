import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

# Exercitiul 2


def high_freq_attenuation(X, freq_cutoff):
    Y = np.fft.fft2(X)
    Y_cutoff = Y.copy()

    freq_db = 20 * np.log10(abs(Y + 1e-15))
    Y_cutoff[freq_db > freq_cutoff] = 0

    return np.real(np.fft.ifft2(Y_cutoff))


SNR = np.inf
SNR_threshold = 0.0072
freq_cuttof = 300

X = misc.face(gray=True)
X_cutoff = misc.face(gray=True)

while SNR > SNR_threshold:
    X_compressed = high_freq_attenuation(X_cutoff, freq_cuttof)
    SNR = np.sum(X ** 2) / np.sum(np.abs(X - X_compressed) ** 2)
    freq_cuttof -= 10
    print(SNR)

Y = np.fft.fft2(X)
Y_compressed = np.fft.fft2(X_cutoff)

fig, axs = plt.subplots(2, 2)
freq_db = 20 * np.log10(abs(Y + 1e-15))

# Imaginea originala

axs[0][0].imshow(X, cmap='gray')
axs[0][0].set_title("Imaginea originala")
axs[1][0].imshow(np.fft.fftshift(freq_db), cmap='gray')
axs[1][0].set_title("Spectrul imaginii originale")

# Imaginea comprimata

freq_db = 20 * np.log10(abs(Y_compressed + 1e-15))

axs[0][1].imshow(X_compressed, cmap='gray')
axs[0][1].set_title("Imaginea comprimata")
axs[1][1].imshow(np.fft.fftshift(freq_db), cmap='gray')
axs[1][1].set_title("Spectrul imaginii comprimate")
plt.tight_layout()

plt.show()
