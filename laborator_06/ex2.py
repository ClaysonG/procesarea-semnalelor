import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

N = 100
p_coef = np.random.randint(0, 10, size=N + 1)
q_coef = np.random.randint(0, 10, size=N + 1)

p = np.poly1d(p_coef)
q = np.poly1d(q_coef)

fig, axs = plt.subplots(2, 1, figsize=(10, 7))
axs[0].plot(sig.convolve(p, q), color='green')
axs[0].set_title('Produsul folosind convolutia: inmultirea polinoamelor')

axs[1].plot(np.real(np.fft.ifft(np.fft.fft(p, n=2*N+1)
            * np.fft.fft(q, n=2*N+1))), color='red')
axs[1].set_title('Produsul folosind convolutia: fft')

for ax in axs:
    ax.grid(True)
plt.tight_layout()

plt.show()
